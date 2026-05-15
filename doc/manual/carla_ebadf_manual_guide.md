# CARLA 崩溃修复 — 人工操作说明

> 本文档为人工修改 CARLA 源码的详细说明，包含每处变更的原因解释和完整命令。  
> 如需直接使用 AI 工具完成代码变更，请参考
> [carla_ebadf_vibe_coding.md](carla_ebadf_vibe_coding.md)。  
> 更深层的技术背景请参考
> [carla_ebadf_fix_complete.md](carla_ebadf_fix_complete.md)。

---

## 崩溃现象与根因

SafeBench 连续运行多个 episode（通常第 4–5 个）时，CARLA 发生崩溃：

- **SIGSEGV**：rpclib TCP socket 被多条线程同时 `close()` → 产生 EBADF 错误 → 传播至 `carla::throw_exception` → `UE_LOG(Fatal)` → 段错误
- **SIGABRT**：UE4 使用 `-fno-exceptions` 编译，`throw` 语句到达时触发 `__cxxabiv1::failed_throw → abort()`

根本原因：每个 episode 创建 3 个传感器（collision/lidar/camera），每个传感器对应一个 rpclib streaming session，4 个 episode ≈ 12 个 session，TCP socket 在多条 ASIO strand 上被重复 close。

---

## 环境变量约定

在所有操作中使用以下变量（根据本机实际路径修改）：

```bash
# CARLA 源码根目录，包含 Makefile 和 Unreal/ 子目录
# 每台安装了 CARLA 的设备应已设置此变量，例如：/home/<user>/Carla/carla
# 若未设置请手动指定：export CARLA_UE4_ROOT=/path/to/carla

# UE4 引擎根目录，需根据本机路径手动设置
# 例如：/home/<user>/UnrealEngine/UnrealEngine_4.26
export UE4_ROOT=<your_ue4_root>
```

---

## 修复一：`Carla.cpp` — EBADF 降级为线程退出

### 位置

```
$CARLA_UE4_ROOT/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Carla.cpp
```

### 原因

`carla::throw_exception` 是 rpclib 内部异常的统一出口。原实现对所有错误都调用 `UE_LOG(Fatal)` 导致进程崩溃；或者尝试 `throw`，但 UE4 以 `-fno-exceptions` 编译，`throw` 在此环境下会触发 `__cxxabiv1::failed_throw → std::abort() → SIGABRT`。

修复策略：识别 EBADF（"Bad file descriptor"）这类 double-close 竞态，**不再使用 `pthread_exit`，改用 `std::terminate()` 干净终止**。

> ⚠️ **第三轮修正（2026-05-11）**：`pthread_exit(nullptr)` 只杀当前 ASIO worker 线程，不调用 C++ 析构器，导致 epoll reactor 的 `descriptor_state` 对象残留在内存中成为野指针。当新连接通过 `start_accept()` 触发 `deregister_descriptor(descriptor_state*&)` 时，解引用无效指针 → 访问 `0x0000000000000090` → **SIGSEGV**。
>
> 正确做法：`close_socket_once()`（修复二）从源头消除双 close。如果 EBADF 仍到达 `throw_exception`，说明原子 close 未生效，此时用 `std::terminate()` 干净终止（SIGABRT），而不是留下腐烂的 epoll 状态。

### 操作步骤

1. 打开文件 `$CARLA_UE4_ROOT/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Carla.cpp`

2. **移除** `#include <pthread.h>`（如果存在）

3. 找到 `throw_exception` 函数（搜索 `void throw_exception`），将整个函数替换为：
   ```cpp
   namespace carla {
     void throw_exception(const std::exception &e) {
       const char* msg = e.what();

       if (msg && std::strstr(msg, "close: Bad file descriptor") != nullptr) {
         UE_LOG(LogCarla, Error,
           TEXT("FATAL: rpclib double-close (EBADF) detected — "
                "close_socket_once() should have prevented this. "
                "Terminating cleanly: %s"),
           UTF8_TO_TCHAR(msg));
         // std::terminate() → SIGABRT：干净终止。
         // 切勿使用 pthread_exit()：它会破坏 epoll 状态，
         // 导致后续 deregister_descriptor 访问 0x90 → SIGSEGV。
         std::terminate();
       }

       UE_LOG(LogCarla, Fatal, TEXT("Exception thrown: %s"), UTF8_TO_TCHAR(msg));
       std::terminate();
     }
   } // namespace carla
   ```

4. 清理旧的编译缓存（否则 `make launch` 可能使用旧 `.o` 文件跳过重编）：
   ```bash
   rm -rf "$CARLA_UE4_ROOT/Unreal/CarlaUE4/Plugins/Carla/Intermediate/Build/Linux/B4D820EA/UE4Editor/Development/Carla/"*.o
   ```

---

## 修复二：`async_writer.h` — 原子 close-once + 退出保护

### 位置

同时修改以下两个文件（内容相同）：

```
$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libcxx-install/include/rpc/detail/async_writer.h
$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libstdcxx-install/include/rpc/detail/async_writer.h
```

### 原因

rpclib TCP socket 有三条并发 close 路径：
1. `do_read()` 的 read strand（连接断开时）
2. `close()` 触发的 write strand lambda
3. `do_write()` 回调中的退出分支

三条路径都调用 `socket_.close()`，但没有任何互斥，导致同一个 fd 被重复 close → EBADF。

修复策略：
- 新增 `std::atomic_bool socket_closed_` 旗标，通过 `compare_exchange_strong` 保证 close 只执行一次（`close_socket_once()`）
- `do_write()` 回调中，用 `if (!exit_)` 保护队列操作，避免 exit_ 后 pop 已清空的队列

### 操作步骤

1. 打开文件，找到 `protected:` 块：
   ```cpp
   protected:
       std::atomic_bool exit_{false};
   ```

2. 在 `exit_` 成员之后、其他成员之前，新增：
   ```cpp
       std::atomic_bool socket_closed_{false};

       // 原子 close-once：三条并发 close 路径只有第一条真正关闭 fd
       void close_socket_once() {
           bool expected = false;
           if (!socket_closed_.compare_exchange_strong(expected, true)) {
               return; // 已被其他 strand 关闭，直接返回
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

3. 找到 `do_write()` 中 `write_strand_.wrap(...)` 的 lambda，找到其中处理 `ec` 和 `exit_` 的部分，修改为：
   ```cpp
   write_strand_.wrap(
       [this, self](std::error_code ec, std::size_t transferred) {
           (void)transferred;
           // exit_=true 时不操作队列，避免与 socket 关闭竞态
           if (!exit_) {
               if (!ec) {
                   write_queue_.pop_front();
                   if (write_queue_.size() > 0) { do_write(); }
               } else {
                   LOG_ERROR("Error while writing to socket: {}", ec);
               }
           }

           if (exit_) {
               close_socket_once();   // 替换原来的 socket_.close() 或 shutdown+close
           }
       }));
   ```

---

## 修复三：`server_session.cc` — 统一 close 路径

`server_session.cc` 在 rpclib 静态库（`librpc.a`）中，修改后需要重新编译并替换库中的 object。

### 步骤一：获取 rpclib 源码

```bash
git clone -b v2.2.1_c5 --depth=1 \
    https://github.com/carla-simulator/rpclib.git /tmp/rpclib_src
```

### 步骤二：修改 `/tmp/rpclib_src/lib/rpc/detail/server_session.cc`

**修改 `close()` 函数**（将 `write_strand_.post` lambda 中的 `socket_.close()` 改为 `close_socket_once()`）：

原代码：
```cpp
void server_session::close() {
    auto self(shared_from_base<server_session>());
    LOG_INFO("Closing session.");
    exit_ = true;
    write_strand_.post([this, self]() {
        socket_.close();
        if (parent_)
            parent_->close_session(self);
    });
}
```

修改为：
```cpp
void server_session::close() {
    auto self(shared_from_base<server_session>());
    LOG_INFO("Closing session.");
    exit_ = true;
    write_strand_.post([this, self]() {
        close_socket_once();
        if (parent_)
            parent_->close_session(self);
    });
}
```

**修改 `do_read()` 末尾的关闭守卫**：

原代码：
```cpp
if (exit_) {
    socket_.close();
}
```

修改为：
```cpp
if (exit_) {
    close_socket_once();
}
```

### 步骤三：编译 server_session.cc

> ⚠️ **重要**：编译时绝对不能加 `-fno-exceptions`！  
> rpclib 头文件 `rpc/this_handler.inl` 含有 `throw` 语句，添加 `-fno-exceptions` 会导致编译失败或运行时 abort。

设置编译变量（与 CARLA 构建系统保持一致）：

```bash
CLANG="$UE4_ROOT/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v17_clang-10.0.1-centos7/x86_64-unknown-linux-gnu/bin/clang++"
SYSROOT="$UE4_ROOT/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v17_clang-10.0.1-centos7/x86_64-unknown-linux-gnu"
LIBCXX="$UE4_ROOT/Engine/Source/ThirdParty/Linux/LibCxx/include/c++/v1"
RPCLIB_INC="$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libcxx-install/include"
BOOST_INC="$CARLA_UE4_ROOT/Build/boost-1.84.0-c10-install/include"
```

执行编译：

```bash
"$CLANG" -std=c++14 -fPIC -stdlib=libc++ \
  -isystem "$LIBCXX" --sysroot="$SYSROOT" \
  -DASIO_NO_EXCEPTIONS -DBOOST_NO_EXCEPTIONS \
  -DRPCLIB_ASIO=clmdep_asio -DRPCLIB_MSGPACK=clmdep_msgpack \
  -DRPCLIB_FMT=clmdep_fmt -DRPCLIB_COMPILE_LIBRARY \
  -I"$RPCLIB_INC" -I/tmp/rpclib_src/dependencies/include \
  -isystem "$BOOST_INC" -w \
  -c /tmp/rpclib_src/lib/rpc/detail/server_session.cc \
  -o /tmp/server_session.cc.o
```

### 步骤四：替换 librpc.a 中的旧 object

两个 librpc.a（libcxx 和 libstdcxx 版本）都需要更新：

```bash
for LIB in \
  "$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libcxx-install/lib/librpc.a" \
  "$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libstdcxx-install/lib/librpc.a"; do
  # 备份原始库
  cp "$LIB" "${LIB}.bak"
  # 删除旧 object
  ar d "$LIB" server_session.cc.o
  # 插入新 object
  ar r "$LIB" /tmp/server_session.cc.o
done
```

命令说明：
- `ar d`：从静态库中删除指定的 object 文件
- `ar r`：向静态库中添加 object 文件（若同名则覆盖）

---

## 最终构建

完成以上所有代码变更后，重新构建并启动 CARLA：

```bash
cd "$CARLA_UE4_ROOT"
make launch
```

`make launch` 会自动完成以下步骤：
1. 重编 `LibCarla.server.release`（包含修改后的 `Carla.cpp`）
2. 重建 Carla UE4 插件（链接更新后的 `librpc.a`）
3. 启动 CARLA Editor

---

## SafeBench Python 侧（已包含在仓库中）

以下 Python 修复**已在本仓库中**，克隆仓库后无需手动修改：

| 文件 | 修复内容 |
|------|---------|
| `safebench/gym_carla/envs/carla_env_tcp.py` | 传感器顺序初始化（Collision → Lidar → Camera），每个等待首帧数据后再创建下一个 |
| `safebench/carla_runner.py` | 每个 episode 结束 `clean_up()` 后 `sleep(5)` 秒，给 CARLA 清理 streaming socket 时间 |

---

## 验证方法

```bash
# 1. 停掉所有旧 CARLA 进程
pkill -f CarlaUE4

# 2. 从修复后的源码构建并启动
cd "$CARLA_UE4_ROOT"
make launch

# 3. 另一终端运行 SafeBench 多 episode
cd /path/to/GUISafeBenchHK
python scripts/run.py --mode eval --agent behavior --scenario standard --seed 0
```

预期效果：运行 20+ episode 不出现 SIGSEGV/SIGABRT，不卡死。

---

## 修复层次汇总

| 层次 | 文件 | 机制 | 效果 |
|------|------|------|------|
| 治本 | `async_writer.h` + `server_session.cc` | `close_socket_once()` 原子旗标 | 三条并发 close 路径只有一条执行，**消除双 close** |
| 治标 | `Carla.cpp` | `std::terminate()` 替代 `pthread_exit` | EBADF 若仍到达 → 干净 SIGABRT，不破坏 epoll 状态 |
| 防御 A | `carla_env_tcp.py` | 传感器顺序初始化 | 减少 rpc session 并发初始化窗口 |
| 防御 B | `carla_runner.py` | Episode 间隔 5s | 给 CARLA 完整清理时间 |
