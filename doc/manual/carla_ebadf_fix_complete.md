# CARLA 0.9.13 双 close 竞态崩溃完整修复方案

> 最后更新：2026-05-06  
> 修复范围：CARLA C++ 源码 (Carla.cpp, rpclib) + SafeBench Python 防御

> **路径约定**：本文档中的所有路径均使用以下环境变量，请在执行命令前设置：
> - `$CARLA_UE4_ROOT`：CARLA 源码根目录（含 `Makefile` 和 `Unreal/` 子目录），每台安装了 CARLA 的设备应已设置此变量，例如 `/home/<user>/Carla/carla`
> - `$UE4_ROOT`：UE4 引擎根目录，需根据本机路径手动设置，例如 `/home/<user>/UnrealEngine/UnrealEngine_4.26`

---

## 问题描述

SafeBench 多 episode 连续运行时（通常在第 4–5 episode），CARLA 进程发生崩溃。

| 崩溃信号 | 根因 |
|----------|------|
| **SIGSEGV** | EBADF 双 close 到达 `carla::throw_exception` → `UE_LOG(Fatal)` |
| **SIGABRT** | `throw std::runtime_error` 在 UE4 的 `-fno-exceptions` 环境下触发 `__cxxabiv1::failed_throw → abort()` |
| 卡死 | 修复后 ASIO worker 线程被 `pthread_exit` 杀死，io_service 挂起 |

**触发链**：每 episode 创建 3 个传感器 (collision/lidar/camera) → 3 个 rpc streaming session → 4 episodes ≈12 sessions → 概率 100%。

---

## 三层并发 close 路径

rpclib 的 TCP socket 被 **3 个不同 strand 上的代码路径** 尝试 close：

```
  read_strand_ (do_read 末尾)
       │
       ├── close(fd)
       │
  write_strand_ (close strand lambda)
       │
       ├── close(fd)   ← 第二次 close → EBADF
       │
  write_strand_ (do_write 退出分支)
       │
       └── close(fd)
```

---

## 修复 1：Carla.cpp — `pthread_exit` 最后防线

**文件**：`Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Carla.cpp`

```cpp
#include <pthread.h> // for pthread_exit    ← 新增
// 移除：#include <stdexcept>

namespace carla {
  void throw_exception(const std::exception &e) {
    const char* msg = e.what();

    if (msg && std::strstr(msg, "close: Bad file descriptor") != nullptr) {
      UE_LOG(LogCarla, Warning,
        TEXT("rpclib double-close race (EBADF) — exiting ASIO worker thread cleanly: %s"),
        UTF8_TO_TCHAR(msg));
      // pthread_exit 只杀当前线程，不杀整个进程。
      // throw 在 UE4 -fno-exceptions 下会触发 failed_throw→abort→SIGABRT。
      pthread_exit(nullptr);
    }

    UE_LOG(LogCarla, Fatal, TEXT("Exception thrown: %s"), UTF8_TO_TCHAR(msg));
    std::terminate();
  }
}
```

> **为什么不用 `throw`**：UE4 用 `-fno-exceptions` 编译，任何 `throw` 都会触发 `__cxxabiv1::failed_throw` → `std::abort()` → **SIGABRT 杀死整个进程**。`pthread_exit` 只退出当前 ASIO worker 线程。

---

## 修复 2：async_writer.h — 原子 close-once + 退出保护

**文件**：`Build/rpclib-v2.2.1_c5-c10-libcxx-install/include/rpc/detail/async_writer.h`

### 2a. 新增原子旗标和方法

```cpp
protected:
    std::atomic_bool exit_{false};
    std::atomic_bool socket_closed_{false};   // ← 新增

    // 原子 close-once：三条并发路径只有第一条真正关闭 fd，后续调用 no-op
    void close_socket_once() {
        bool expected = false;
        if (!socket_closed_.compare_exchange_strong(expected, true)) {
            return; // 已被其他 strand 关闭
        }
        LOG_INFO("Closing socket (close_socket_once)");
        std::error_code e;
        socket_.shutdown(RPCLIB_ASIO::ip::tcp::socket::shutdown_both, e);
        if (e) {
            LOG_WARN("std::system_error during socket shutdown. "
                     "Code: {}. Message: {}", e.value(), e.message());
        }
        std::error_code close_ec;
        socket_.close(close_ec);
        if (close_ec) {
            LOG_WARN("Error while closing socket in close_socket_once "
                     "(code {}): {}", close_ec.value(), close_ec.message());
        }
    }
```

### 2b. do_write() 退出保护

```cpp
void do_write() {
    // ... async_write ...

    write_strand_.wrap(
        [this, self](std::error_code ec, std::size_t transferred) {
            (void)transferred;
            // exit_=true 时跳过队列操作，sbuffer 由 deque 析构安全释放
            if (!exit_) {
                if (!ec) {
                    write_queue_.pop_front();
                    if (write_queue_.size() > 0) { do_write(); }
                } else {
                    LOG_ERROR("Error while writing to socket: {}", ec);
                }
            }

            if (exit_) {
                close_socket_once();   // ← 替换原先的 shutdown+close
            }
        }));
}
```

**同步到两个 rpclib 安装**：
```bash
# libcxx 版本（主要使用）
cp async_writer.h .../rpclib-v2.2.1_c5-c10-libcxx-install/include/rpc/detail/async_writer.h
# libstdcxx 版本（备选）
cp async_writer.h .../rpclib-v2.2.1_c5-c10-libstdcxx-install/include/rpc/detail/async_writer.h
```

---

## 修复 3：server_session.cc — 三条 close 路径统一

**文件**：`/tmp/rpclib_src/lib/rpc/detail/server_session.cc`  
（rpclib 源码，clone 自 `https://github.com/carla-simulator/rpclib.git -b v2.2.1_c5`）

### 3a. close() strand lambda

```cpp
void server_session::close() {
    auto self(shared_from_base<server_session>());
    LOG_INFO("Closing session.");
    exit_ = true;
    write_strand_.post([this, self]() {
        close_socket_once();               // ← 替换 socket_.close()
        if (parent_)
            parent_->close_session(self);
    });
}
```

### 3b. do_read() 末尾守卫

```cpp
void server_session::do_read() {
    // ... async_read_some + handler ...

    if (exit_) {
        close_socket_once();               // ← 替换 socket_.close()
    }
}
```

---

## 修复 4：编译 & 替换 librpc.a

> **注意**：编译时不能使用 `-fno-exceptions`，因为 rpclib 头文件 `this_handler.inl` 含有 `throw` 语句。

```bash
CLANG="$UE4_ROOT/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v17_clang-10.0.1-centos7/x86_64-unknown-linux-gnu/bin/clang++"
SYSROOT="$UE4_ROOT/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v17_clang-10.0.1-centos7/x86_64-unknown-linux-gnu"
LIBCXX="$UE4_ROOT/Engine/Source/ThirdParty/Linux/LibCxx/include/c++/v1"
RPCLIB_INC="$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libcxx-install/include"
BOOST_INC="$CARLA_UE4_ROOT/Build/boost-1.84.0-c10-install/include"

$CLANG -std=c++14 -fPIC -stdlib=libc++ \
  -isystem "$LIBCXX" --sysroot="$SYSROOT" \
  -DASIO_NO_EXCEPTIONS -DBOOST_NO_EXCEPTIONS \
  -DRPCLIB_ASIO=clmdep_asio -DRPCLIB_MSGPACK=clmdep_msgpack \
  -DRPCLIB_FMT=clmdep_fmt -DRPCLIB_COMPILE_LIBRARY \
  -I"$RPCLIB_INC" -I/tmp/rpclib_src/dependencies/include \
  -isystem "$BOOST_INC" -w \
  -c /tmp/rpclib_src/lib/rpc/detail/server_session.cc \
  -o /tmp/server_session.cc.o

# 替换两个 librpc.a
for LIB in \
  "$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libcxx-install/lib/librpc.a" \
  "$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libstdcxx-install/lib/librpc.a"; do
  ar d "$LIB" server_session.cc.o
  ar r "$LIB" /tmp/server_session.cc.o
done
```

---

## 修复 5：重建 CARLA

```bash
cd /home/fsm/Carla/carla

# 清理 unity build 缓存（关键！否则旧代码仍在 .o 中）
rm -rf Unreal/CarlaUE4/Plugins/Carla/Intermediate/Build/Linux/B4D820EA/UE4Editor/Development/Carla/*.o

# 全量重建 + 启动
make launch
```

> `make launch` 会依次执行：
> - `LibCarla.server.release` — 重编服务端 C++ 库
> - `BuildUE4Plugins.sh --build`
> - `BuildCarlaUE4.sh --build --launch` — 重建 Carla 插件 **并启动**

---

## 修复 6：SafeBench — 传感器顺序初始化

**文件**：`safebench/gym_carla/envs/carla_env_tcp.py`

`_attach_sensor()` 改为 Collision → Lidar → Camera 顺序生成，每个等待首帧数据后再生成下一个：

```python
def _attach_sensor(self):
    # 1. Collision (无数据 sentinel，tick 一次)
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, ...)
    self.collision_sensor.listen(get_collision_hist)
    self.world.tick()

    # 2. Lidar (等待第一帧)
    self.lidar_data = None
    self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, ...)
    self.lidar_sensor.listen(get_lidar_data)
    self._wait_sensor_first_data('lidar_data')

    # 3. Camera (等待第一帧)
    self.camera_img = None
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, ...)
    self.camera_sensor.listen(get_camera_img)
    self._wait_sensor_first_data('camera_img')
```

---

## 修复 7：SafeBench — Episode 间隔 5 秒

**文件**：`safebench/carla_runner.py`

两处各加 5 秒 sleep，给 CARLA 清理 streaming socket 的时间：

```python
# 训练循环 (line ~214)
self.env.clean_up()
time.sleep(5)  # ← 新增：等待 CARLA 清理

# 评估循环 (line ~370)
self.env.clean_up()
time.sleep(5)  # ← 新增：等待 CARLA 清理
```

---

## 修复效果总结

| 层次 | 机制 | 效果 |
|------|------|------|
| 治本 | `close_socket_once()` 原子旗标 | 三条并发 close 路径只有一条执行，**消除双 close** |
| 治标 | `pthread_exit` 替代 `throw` | 若 EBADF 仍到达 → 只杀 worker 线程，进程存活 |
| 防御 A | 传感器顺序初始化 | 减少 rpc session 初始化时的并发窗口 |
| 防御 B | Episode 5s 间隔 | 给 CARLA 完整清理时间 |

---

## 验证方法

```bash
# 1. 停掉所有旧 CARLA 进程
pkill -f CarlaUE4

# 2. 从源码重建 + 启动
cd "$CARLA_UE4_ROOT"
make launch

# 3. 在另一个终端运行 SafeBench
cd /path/to/GUISafeBenchHK
python scripts/run.py --mode eval --agent behavior --scenario standard --seed 0
```

预期：运行 20+ episodes 不崩溃不卡死。
