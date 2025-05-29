#include "logger.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h> // 控制台输出
#include <spdlog/sinks/daily_file_sink.h> // 每日文件输出

void setup_logs() {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    sinks.push_back(
        std::make_shared<spdlog::sinks::daily_file_sink_st>("logs/logfile.log", 23, 59));
    // create synchronous  loggers
    auto ldpc_log = std::make_shared<spdlog::logger>("LDPC", begin(sinks), end(sinks));
    auto hw_logger = std::make_shared<spdlog::logger>("GPU", begin(sinks), end(sinks));
    auto results_logger = std::make_shared<spdlog::logger>("RESULTS", begin(sinks), end(sinks));
    auto debug_logger = std::make_shared<spdlog::logger>("DEBUG", begin(sinks), end(sinks));
    spdlog::register_logger(ldpc_log);
    spdlog::register_logger(hw_logger);
    spdlog::register_logger(results_logger);
    spdlog::register_logger(debug_logger);
}