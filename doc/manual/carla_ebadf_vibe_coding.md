# CARLA 崩溃修复 — Vibe Coding 代码变更说明

> 本文档供 AI 编码工具（Vibe Coding / Copilot Agent 等）直接使用。  
> 请严格按照下列指示修改对应文件，代码修改完成后由**人工执行 `make launch`** 完成构建。  
> SafeBench Python 侧（`carla_env_tcp.py`、`carla_runner.py`）的修复已包含在本仓库中，**无需再次修改**。

---

## 环境变量约定

在执行任何命令前，确认以下环境变量已设置：

```bash
# CARLA 源码根目录（包含 Makefile 和 Unreal/ 目录）
# 每台安装了 CARLA 的设备应已设置此变量，例如：/home/<user>/Carla/carla
# 若未设置请手动指定：export CARLA_UE4_ROOT=/path/to/carla

# UE4 引擎根目录，需根据本机路径手动设置
# 例如：/home/<user>/UnrealEngine/UnrealEngine_4.26
export UE4_ROOT=<your_ue4_root>
```

所有后续路径均基于这两个变量。

---

## 变更 1：`Carla.cpp` — EBADF 降级处理

**文件路径**：
```
$CARLA_UE4_ROOT/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Carla.cpp
```

**变更说明**：将 `throw_exception` 函数中对 EBADF（"Bad file descriptor"）的处理从 `UE_LOG(Fatal)` 改为 `pthread_exit`，避免整个进程被杀死。

**在文件顶部 include 区域新增一行**（加在其他 `#include` 之后）：

```cpp
#include <pthread.h>
```

**找到 `throw_exception` 函数，将其完整替换为以下内容**：

```cpp
namespace carla {
  void throw_exception(const std::exception &e) {
    const char* msg = e.what();

    if (msg && std::strstr(msg, "close: Bad file descriptor") != nullptr) {
      UE_LOG(LogCarla, Warning,
        TEXT("rpclib double-close race (EBADF) — exiting ASIO worker thread cleanly: %s"),
        UTF8_TO_TCHAR(msg));
      // pthread_exit 只退出当前 ASIO worker 线程，不杀整个进程。
      // 不能用 throw：UE4 以 -fno-exceptions 编译，throw 会触发 abort() → SIGABRT。
      pthread_exit(nullptr);
    }

    UE_LOG(LogCarla, Fatal, TEXT("Exception thrown: %s"), UTF8_TO_TCHAR(msg));
    std::terminate();
  }
} // namespace carla
```

**变更后，清理旧编译缓存**（执行以下 shell 命令）：

```bash
rm -rf "$CARLA_UE4_ROOT/Unreal/CarlaUE4/Plugins/Carla/Intermediate/Build/Linux/B4D820EA/UE4Editor/Development/Carla/"*.o
```

---

## 变更 2：`async_writer.h` — 原子 close-once + 退出保护

**文件路径（libcxx 版本，主用）**：
```
$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libcxx-install/include/rpc/detail/async_writer.h
```

**同时修改 libstdcxx 版本（备用）**：
```
$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libstdcxx-install/include/rpc/detail/async_writer.h
```

两个文件改动相同。

### 2a. 在 `protected:` 块中新增原子旗标和 `close_socket_once()` 方法

**找到**：
```cpp
protected:
    std::atomic_bool exit_{false};
```

**替换为**：
```cpp
protected:
    std::atomic_bool exit_{false};
    std::atomic_bool socket_closed_{false};

    // 原子 close-once：三条并发 close 路径只有第一条真正关闭 fd，后续 no-op
    void close_socket_once() {
        bool expected = false;
        if (!socket_closed_.compare_exchange_strong(expected, true)) {
            return;
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

### 2b. 修改 `do_write()` 的写完成回调

**找到** `write_strand_.wrap(...)` 的 lambda 中原有的队列操作和退出分支：

```cpp
write_strand_.wrap(
    [this, self](std::error_code ec, std::size_t transferred) {
        (void)transferred;
        if (!ec) {
            write_queue_.pop_front();
            if (write_queue_.size() > 0) { do_write(); }
        } else {
            LOG_ERROR("Error while writing to socket: {}", ec);
        }

        if (exit_) {
            // 原来这里是：socket_.close() 或 shutdown + close
        }
    }));
```

**替换为**（注意：用 `if (!exit_)` 包裹队列操作，将 close 路径改为 `close_socket_once()`）：

```cpp
write_strand_.wrap(
    [this, self](std::error_code ec, std::size_t transferred) {
        (void)transferred;
        // exit_=true 时跳过队列操作，sbuffer 由 deque 析构安全释放，
        // 避免 pop_front() 与 socket_.close() 之间的堆损坏竞态。
        if (!exit_) {
            if (!ec) {
                write_queue_.pop_front();
                if (write_queue_.size() > 0) { do_write(); }
            } else {
                LOG_ERROR("Error while writing to socket: {}", ec);
            }
        }

        if (exit_) {
            close_socket_once();
        }
    }));
```

---

## 变更 3：`server_session.cc` — 编译并替换 librpc.a

`server_session.cc` 位于 rpclib 源码中，需要 clone 源码、修改、编译，再替换 `librpc.a` 中的 object。

### 3a. Clone rpclib 源码

```bash
git clone -b v2.2.1_c5 --depth=1 \
    https://github.com/carla-simulator/rpclib.git /tmp/rpclib_src
```

### 3b. 修改 `/tmp/rpclib_src/lib/rpc/detail/server_session.cc`

**修改 `close()` 函数**：

找到：
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

替换为：
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

**修改 `do_read()` 末尾的 close 调用**：

找到 `do_read()` 中的：
```cpp
if (exit_) {
    socket_.close();
}
```

替换为：
```cpp
if (exit_) {
    close_socket_once();
}
```

### 3c. 编译 server_session.cc

```bash
CLANG="$UE4_ROOT/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v17_clang-10.0.1-centos7/x86_64-unknown-linux-gnu/bin/clang++"
SYSROOT="$UE4_ROOT/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v17_clang-10.0.1-centos7/x86_64-unknown-linux-gnu"
LIBCXX="$UE4_ROOT/Engine/Source/ThirdParty/Linux/LibCxx/include/c++/v1"
RPCLIB_INC="$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libcxx-install/include"
BOOST_INC="$CARLA_UE4_ROOT/Build/boost-1.84.0-c10-install/include"

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

> ⚠️ **注意**：此处不能加 `-fno-exceptions`，因为 rpclib 头文件 `this_handler.inl` 含有 `throw` 语句。

### 3d. 替换两个 librpc.a 中的 object

```bash
for LIB in \
  "$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libcxx-install/lib/librpc.a" \
  "$CARLA_UE4_ROOT/Build/rpclib-v2.2.1_c5-c10-libstdcxx-install/lib/librpc.a"; do
  cp "$LIB" "${LIB}.bak"
  ar d "$LIB" server_session.cc.o
  ar r "$LIB" /tmp/server_session.cc.o
done
```

---

## 人工操作：构建 CARLA

> **以上代码变更完成后，由人工执行以下命令**：

```bash
cd "$CARLA_UE4_ROOT"
make launch
```

`make launch` 会：
1. 重新编译 LibCarla（包含修改后的 `Carla.cpp`）
2. 链接更新后的 `librpc.a`
3. 重建 Carla UE4 插件并启动 CARLA Editor

---

## 变更汇总

| 文件 | 操作 | 机制 |
|------|------|------|
| `Carla.cpp` | 修改 `throw_exception()`，EBADF 改为 `pthread_exit` | 治标：EBADF 不再杀进程 |
| `async_writer.h` | 新增 `close_socket_once()`，`do_write()` 加退出保护 | 治本：三条并发 close 路径原子化 |
| `server_session.cc` | `close()` 和 `do_read()` 改用 `close_socket_once()` | 治本：统一 close 入口 |
| 构建 | `make launch` | 将以上修改编译进 CARLA |

> SafeBench Python 侧（传感器顺序初始化、episode 间隔 5s）已在本仓库中，克隆即生效，无需额外操作。
