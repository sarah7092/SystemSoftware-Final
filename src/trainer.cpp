/// @file trainer.cpp
/// @brief Implementation of the trainer (orchestrator) executable.
#include "trainer.hpp"

#include <csignal>
#include <cstring>
#include <fcntl.h>     // for fcntl, FD_CLOEXEC
#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
    volatile std::sig_atomic_t g_child_exited = 0;

    void sigchld_handler(int)
    {
        // TODO: mark that at least one child process has exited.
        // Hint: set the global flag g_child_exited to a non-zero value.
        //
        // This flag can be used in the main loop to notice that a SIGCHLD
        // was delivered and then call wait()/waitpid() to reap children.
        g_child_exited = 1;
    }

    void set_cloexec(int fd)
    {
        // TODO: mark this file descriptor as close-on-exec using fcntl().
        //
        // 1. Use fcntl(fd, F_GETFD) to get current flags.
        // 2. If the call succeeds, OR the result with FD_CLOEXEC.
        // 3. Use fcntl(fd, F_SETFD, new_flags) to update.
        //
        // This prevents child processes (after exec) from inheriting
        // these pipe descriptors unintentionally.
        
        int flags = fcntl(fd, F_GETFD);
    
        if (flags != -1) {
            int new_flags = flags | FD_CLOEXEC;

            if (fcntl(fd, F_SETFD, new_flags) == -1) {
                std::perror("fcntl F_SETFD");
            }
        } else {
            std::perror("fcntl F_GETFD");
        }
    }

    int spawn_child(const char *prog,
                    char *const argv[],
                    int stdin_fd,
                    int stdout_fd)
    {
        // TODO: fork a child process, hook up its stdin/stdout, and exec 'prog'.
        //
        // Required behavior:
        //   - Call fork().
        //   - On error (pid < 0), print an error with std::perror("fork")
        //     and return -1.
        //
        //   - In the child (pid == 0):
        //       * If stdin_fd >= 0 and stdin_fd != STDIN_FILENO,
        //         dup2(stdin_fd, STDIN_FILENO); on error, perror and _exit(1).
        //       * If stdout_fd >= 0 and stdout_fd != STDOUT_FILENO,
        //         dup2(stdout_fd, STDOUT_FILENO); on error, perror and _exit(1).
        //       * Call execvp(prog, argv).
        //         On error, print with std::perror("execvp") and _exit(1).
        //
        //   - In the parent:
        //       * Simply return the child's PID (the value from fork()).
        //
        // This function does NOT close any file descriptors; the caller
        // (trainer::run) remains responsible for closing unused pipe ends.

        pid_t pid = fork();

        if (pid < 0) {
            std::perror("fork");
            return -1;
        }

        if (pid == 0) {
            if (stdin_fd >= 0 && stdin_fd != STDIN_FILENO) {
                if (dup2(stdin_fd, STDIN_FILENO) < 0) {
                    std::perror("dup2 stdin");
                    _exit(1);
                }
            }

            if (stdout_fd >= 0 && stdout_fd != STDOUT_FILENO) {
                if (dup2(stdout_fd, STDOUT_FILENO) < 0) {
                    std::perror("dup2 stdout");
                    _exit(1);
                }
            }
            
            execvp(prog, argv);
            
            std::perror("execvp");
            _exit(1);
        }
        
        return static_cast<int>(pid);
    }


} // namespace

namespace trainer
{
    int run(const std::string &csv_path)
    {
        // Install SIGCHLD handler so we can notice if a child dies unexpectedly.
        std::signal(SIGCHLD, sigchld_handler);

        int pipe_pre_to_fwd[2];
        int pipe_fwd_to_bwd[2];
        int pipe_bwd_to_log[2];

        if (pipe(pipe_pre_to_fwd) < 0)
        {
            std::perror("pipe pre->fwd");
            return 1;
        }
        if (pipe(pipe_fwd_to_bwd) < 0)
        {
            std::perror("pipe fwd->bwd");
            return 1;
        }
        if (pipe(pipe_bwd_to_log) < 0)
        {
            std::perror("pipe bwd->log");
            return 1;
        }

        // Mark all pipe FDs as close-on-exec. Children that need them will
        // dup2 them onto stdin/stdout before exec; the dup'd FDs (0/1) will NOT
        // have FD_CLOEXEC, so they survive exec, while the originals are closed.
        set_cloexec(pipe_pre_to_fwd[0]);
        set_cloexec(pipe_pre_to_fwd[1]);
        set_cloexec(pipe_fwd_to_bwd[0]);
        set_cloexec(pipe_fwd_to_bwd[1]);
        set_cloexec(pipe_bwd_to_log[0]);
        set_cloexec(pipe_bwd_to_log[1]);

        // Copy CSV path into a mutable buffer for argv.
        char csv_arg[1024];
        std::strncpy(csv_arg, csv_path.c_str(), sizeof(csv_arg) - 1);
        csv_arg[sizeof(csv_arg) - 1] = '\0';

        // Executable paths (built into bin/ by build.sh)
        const char *pre_prog = "bin/preprocess";
        const char *fwd_prog = "bin/forward_layer";
        const char *bwd_prog = "bin/backward_layer";
        const char *log_prog = "bin/logger";

        // argv arrays (argv[0] should be the program name/path)
        char *pre_argv[] = {
            const_cast<char *>(pre_prog),
            csv_arg,
            nullptr
        };

        char *fwd_argv[] = {
            const_cast<char *>(fwd_prog),
            nullptr
        };

        char *bwd_argv[] = {
            const_cast<char *>(bwd_prog),
            nullptr
        };

        char *log_argv[] = {
            const_cast<char *>(log_prog),
            nullptr
        };

        // Spawn children with appropriate pipe ends for stdin/stdout.
        pid_t pre_pid = spawn_child(pre_prog, pre_argv,
                                    /*stdin_fd=*/-1,
                                    /*stdout_fd=*/pipe_pre_to_fwd[1]);
        if (pre_pid < 0) return 1;

        pid_t fwd_pid = spawn_child(fwd_prog, fwd_argv,
                                    /*stdin_fd=*/pipe_pre_to_fwd[0],
                                    /*stdout_fd=*/pipe_fwd_to_bwd[1]);
        if (fwd_pid < 0) return 1;

        pid_t bwd_pid = spawn_child(bwd_prog, bwd_argv,
                                    /*stdin_fd=*/pipe_fwd_to_bwd[0],
                                    /*stdout_fd=*/pipe_bwd_to_log[1]);
        if (bwd_pid < 0) return 1;

        pid_t log_pid = spawn_child(log_prog, log_argv,
                                    /*stdin_fd=*/pipe_bwd_to_log[0],
                                    /*stdout_fd=*/-1); // logger uses parent's stdout
        if (log_pid < 0) return 1;

        // Parent closes its copies of pipe ends.
        close(pipe_pre_to_fwd[0]);
        close(pipe_pre_to_fwd[1]);
        close(pipe_fwd_to_bwd[0]);
        close(pipe_fwd_to_bwd[1]);
        close(pipe_bwd_to_log[0]);
        close(pipe_bwd_to_log[1]);

        // Wait for children to exit.
        int status = 0;
        pid_t wpid;
        while ((wpid = wait(&status)) > 0)
        {
            if (WIFEXITED(status))
            {
                std::cerr << "trainer: child " << wpid
                          << " exited with status " << WEXITSTATUS(status) << '\n';
            }
            else if (WIFSIGNALED(status))
            {
                std::cerr << "trainer: child " << wpid
                          << " terminated by signal " << WTERMSIG(status) << '\n';
            }
        }

        return 0;
    }

} // namespace trainer

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: trainer <csv_path>\n";
        return 1;
    }
    return trainer::run(argv[1]);
}
