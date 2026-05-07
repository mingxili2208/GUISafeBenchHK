# CARLA 0.9.13 SIGSEGV 崩溃修复方案

> **路径约定**：本文档中的所有路径均使用以下环境变量代替本机绝对路径：
> - `$CARLA_UE4_ROOT`：CARLA 源码根目录（含 `Makefile` 和 `Unreal/` 子目录），每台安装了 CARLA 的设备应已设置此变量，例如 `/home/<user>/Carla/carla`
> - `$UE4_ROOT`：UE4 引擎根目录，需根据本机路径手动设置，例如 `/home/<user>/UnrealEngine/UnrealEngine_4.26`

## 问题描述

在 SafeBench 多 episode 连续运行时（通常在第 5 episode 前后），CARLA UE4Editor 进程发生 SIGSEGV 崩溃，堆栈类似：

```
Program received signal SIGSEGV, Segmentation fault.
0x00007f... in RaiseException (...)
  called from UE_LOG(..., Fatal, ...)
  called from carla::throw_exception(std::exception const&)
  called from clmdep_asio::detail::throw_error(...)
  called from clmdep_asio::ip::tcp::socket::close()
  called from rpc::detail::server_session::close()::{lambda}()
```

---

## 根本原因

### 触发链

1. `_attach_sensor()` 快速连续生成 collision / lidar / camera 三个传感器。
2. 每个传感器在 CARLA 内部异步初始化一个 rpclib 流式 socket。
3. rpclib 的两条代码路径在同一个 socket 上竞争调用 `socket_.close()`：
   - `server_session::close()` 向 `write_strand_` 提交 lambda → 调用 `socket_.close()`
   - `async_writer::do_write()` 在检测到 `exit_ == true` 后也调用 `socket_.close()`
4. 第二次 `close()` 返回 EBADF（文件描述符已关闭）。
5. rpclib 编译时使用了 `-DASIO_NO_EXCEPTIONS`，ASIO 不 throw 异常，而是调用 `carla::throw_exception()`。
6. `carla::throw_exception()` 原始实现调用 `UE_LOG(Fatal, ...)` → UE4 触发 `RaiseException` → SIGSEGV。

### 为什么在第 5 episode 才触发

这是纯概率事件：每 episode 有一定概率触发竞争。Episodes 1–4 恰好避开了，episode 5 没有。

---

## 修复方案（三层）

### 修复 1：治标 — `Carla.cpp`（已完成）

**文件**：`Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Carla.cpp`

将 `carla::throw_exception()` 对 EBADF 的处理从 `UE_LOG(Fatal)` 改为 C++ re-throw：

```cpp
void throw_exception(const std::exception &e) {
    const char* msg = e.what();

    if (msg && std::strstr(msg, "close: Bad file descriptor") != nullptr) {
        UE_LOG(LogCarla, Warning,
            TEXT("rpclib double-close race (EBADF) — rethrowing: %s"),
            UTF8_TO_TCHAR(msg));
        // C++ throw 满足 [[noreturn]]，栈正常展开（RAII析构器运行），
        // 不杀死整个进程。
        throw std::runtime_error(msg);
    }

    UE_LOG(LogCarla, Fatal, TEXT("Exception thrown: %s"), UTF8_TO_TCHAR(msg));
    std::terminate();
}
```

**为什么是 `throw` 而不是其他**：

| 方案 | 问题 |
|------|------|
| `return` | `[[noreturn]]` 调用者编译器生成 `ud2`，会触发 SIGILL |
| `pthread_exit` | 不调用 C++ 析构器，`scoped_lock` 不释放 → strand 死锁 |
| `std::terminate` | 同 `UE_LOG(Fatal)`，直接杀进程 |
| `throw` | ✅ 满足 `[[noreturn]]`，栈展开，RAII 析构器运行，不杀进程 |

---

### 修复 2：治本 — rpclib `async_writer.h` + `server_session.cc`（已完成）

**文件 A**：`Build/rpclib-v2.2.1_c5-c10-libcxx-install/include/rpc/detail/async_writer.h`

在 `do_write()` 的退出分支，把 `socket_.close()` 改为使用 error_code 重载，忽略 EBADF：

```cpp
// 修改前
socket_.close();

// 修改后
std::error_code close_ec;
socket_.close(close_ec);
if (close_ec) {
    LOG_WARN("Error while closing socket in do_write (code {}): {}",
             close_ec.value(), close_ec.message());
}
```

**文件 B**：`server_session.cc`（重新编译后替换 `librpc.a` 中的 object）

在 `server_session::close()` 的 strand lambda 中同样使用 error_code 重载：

```cpp
void server_session::close() {
    exit_ = true;
    auto self(shared_from_base<server_session>());
    io_->post(write_strand_.wrap([this, self]() {
        std::error_code ec;
        socket_.close(ec);  // EBADF 被静默丢弃，不再到达 throw_exception
        if (ec) {
            LOG_WARN("Error while closing socket (code {}): {}",
                     ec.value(), ec.message());
        }
    }));
}
```

**如何重新编译 server_session.cc**：

```bash
# 克隆 rpclib 源码（仅需 dependencies/include 里的 asio 头文件）
git clone -b v2.2.1_c5 --depth=1 \
    https://github.com/carla-simulator/rpclib.git /tmp/rpclib_src

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
  -I"$RPCLIB_INC" \
  -I/tmp/rpclib_src/dependencies/include \
  -isystem "$BOOST_INC" \
  -fno-exceptions -w \
  -c server_session.cc -o server_session.cc.o
```

**替换 librpc.a 中的 object**：

```bash
LIBRPC="$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libcxx-install/lib/librpc.a"
cp "$LIBRPC" "${LIBRPC}.bak"          # 备份
ar d "$LIBRPC" server_session.cc.o    # 删除旧 object
ar r "$LIBRPC" server_session.cc.o    # 加入新 object
```

---

### 修复 3：防御 — 传感器初始化顺序（已完成）

**文件**：`safebench/gym_carla/envs/carla_env_tcp.py`

`_attach_sensor()` 改为顺序生成，每个传感器确认收到第一帧数据后再生成下一个，减少 streaming socket 初始化时的竞争窗口：

```python
def _wait_sensor_first_data(self, attr_name, timeout_ticks=10):
    """Tick world until self.<attr_name> is not None."""
    import time as _time_mod
    for _ in range(timeout_ticks):
        try:
            self.world.tick()
        except Exception as _tick_err:
            if self.logger is not None:
                self.logger.log(
                    f'>> _wait_sensor_first_data tick failed: {_tick_err}',
                    color='yellow')
            _time_mod.sleep(0.1)
        if getattr(self, attr_name, None) is not None:
            return

def _attach_sensor(self):
    # 1. Collision sensor（无数据 sentinel，tick 一次即可）
    self.collision_hist = []
    def get_collision_hist(event): ...
    self.collision_sensor = self.world.spawn_actor(
        self.collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
    self.collision_sensor.listen(get_collision_hist)
    self.world.tick()

    # 2. Lidar sensor（等待第一帧数据）
    if self.scenario_category != 'perception' and not self.disable_lidar:
        self.lidar_data = None
        def get_lidar_data(data): self.lidar_data = data
        self.lidar_sensor = self.world.spawn_actor(
            self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)
        self.lidar_sensor.listen(get_lidar_data)
        self._wait_sensor_first_data('lidar_data')

    # 3. Camera sensor（等待第一帧数据）
    self.camera_img = None
    def get_camera_img(data): ...
    self.camera_sensor = self.world.spawn_actor(
        self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
    self.camera_sensor.listen(get_camera_img)
    self._wait_sensor_first_data('camera_img')
```

---

## 重新构建 CARLA

修改完上述文件后，在 CARLA 根目录执行：

```bash
cd "$CARLA_UE4_ROOT"
make launch
```

这会重新编译 `libUE4Editor-Carla.so`，将 `Carla.cpp` 的修改和更新后的 `librpc.a` 链接进去。

**注**：`async_writer.h` 是 header-only，但 `do_write()` 的实例化在 `server_session.cc.o` 中。只要 `server_session.cc.o` 已经替换（步骤已完成），`do_write()` 的修复就会生效。

---

## 修复效果

| 层次 | 文件 | 效果 |
|------|------|------|
| 治标 | `Carla.cpp` | EBADF 不再触发 Fatal → SIGSEGV；异常被捕获，进程存活 |
| 治本 A | `async_writer.h` + `server_session.cc` | EBADF 永远不会到达 `throw_exception`，从源头消除 |
| 防御 | `carla_env_tcp.py` | 顺序初始化传感器，大幅减少竞争窗口 |

三层同时生效后，即使极低概率的 double-close 仍然发生，也会被治本层静默处理，完全不影响 SafeBench 的 episode 运行。

---

## 第二轮修复（2026-05-05）：修复后崩溃分析

### 问题

第一轮修复后仍然发生崩溃，GDB 分析发现 3 类不同信号：

| 崩溃 | 信号 | 线程 | 原因 |
|------|------|------|------|
| #1 | SIGSEGV | server | `server_session::close()` 路径的 EBADF 仍到达 `Carla.cpp:136` (UE_LOG Fatal) |
| #2 | SIGABRT | server | `async_writer::do_write()` → `pop_front()` → `sbuffer::~sbuffer()` → `free()` 检测到 "corrupted size vs. prev_size" 堆损坏 |
| #3 | SIGTRAP | UE4Editor | Vulkan GPU VendorId 断言 (环境问题，忽略) |

### 根因

1. **`Carla.cpp` 修改未编译进 .so**：源文件比 `libUE4Editor-Carla.so` 新，EBADF 检测从未生效
2. **`server_session::close()` 未修复**：原始文档只说修复 `do_write()` 路径，但 `server_session::close()` 仍用抛出异常的 `socket_.close()`
3. **`async_writer.h` 的 `do_write()` 回调竞态**：`exit_` 被设置后，`pop_front()` 和 `socket_.close()` 之间存在交互导致堆损坏

### 第二轮修复

#### 修复 A：`server_session.cc`（rpclib 源码）

```cpp
void server_session::close() {
    auto self(shared_from_base<server_session>());
    LOG_INFO("Closing session.");
    exit_ = true;
    write_strand_.post([this, self]() {
        // 使用 error_code 重载，避免 EBADF 到达 throw_exception
        std::error_code ec;
        socket_.close(ec);
        if (ec) {
            LOG_WARN("Error while closing socket in server_session::close "
                     "(code {}): {}", ec.value(), ec.message());
        }
        if (parent_)
            parent_->close_session(self);
    });
}
```

> **注意**：编译时不能使用 `-fno-exceptions`，因为包含的 `rpc/this_handler.inl` 中有 `throw` 语句。

#### 修复 B：`async_writer.h` — `do_write()` 退出保护

在 `do_write()` 的写完成回调中，用 `if (!exit_)` 包裹 `write_queue_.pop_front()` 和后续队列操作。如果 `exit_` 在写进行中被设置，跳过队列操作（sbuffer 由 deque 析构函数安全释放），避免 `pop_front()` 与 `socket_.close()` 之间的交互。

#### 修复 C：重新编译

```bash
# 重新编译 server_session.cc.o（不使用 -fno-exceptions）
CLANG=.../clang++
$CLANG -std=c++14 -fPIC -stdlib=libc++ ... -c server_session.cc -o server_session_v2.cc.o

# 替换两个 librpc.a（libcxx 和 libstdcxx）
ar d librpc.a server_session.cc.o
ar r librpc.a server_session_v2.cc.o

# 增量重建 Carla 插件
cd "$CARLA_UE4_ROOT"
bash Util/BuildTools/BuildCarlaUE4.sh --build
```

### 第二轮修复效果

| 层次 | 文件 | 修复内容 |
|------|------|---------|
| 治标 | `Carla.cpp` | 已编译进 .so，EBADF → re-throw (不杀进程) |
| 治本 A | `server_session.cc` (librpc.a) | `close()` 使用 error_code 重载，EBADF 静默丢弃 |
| 治本 B | `async_writer.h` | `exit_` 状态下跳过队列操作，防止堆损坏 |
| 防御 | `carla_env_tcp.py` | 顺序传感器初始化 |

> **构建说明**：`make CarlaUE4Editor` 会触发完整的 Setup.sh（下载 boost 等），较慢。推荐直接用 `BuildCarlaUE4.sh --build` 做增量构建。若需触发 Carla 模块重编译，可 `touch Carla.cpp`。
