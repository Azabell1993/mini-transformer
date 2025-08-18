#include <utils/utils.hpp>
#include <nlohmann/json.hpp> // JSON 파서(헤더 온리)
#include <fstream>
#include <cstdarg>
#include <cstdio>

namespace utils {

// 단순한 로그 출력 유틸리티(표준 에러로 출력)
static void vlog(const char* level, const char* fmt, va_list ap) {
    std::fprintf(stderr, "[%s] ", level);
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
}

void logInfo(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); vlog("INFO", fmt, ap); va_end(ap);
}
void logError(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); vlog("ERROR", fmt, ap); va_end(ap);
}

// 엔진 설정 JSON을 읽어 구조체에 채워 넣습니다.
bool loadEngineConfig(const std::string& path, engine::EngineConfig& out) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    nlohmann::json j; ifs >> j;

    auto jc = j["common"];
    out.common.api_port = jc.value("api_port", out.common.api_port);
    out.common.license = jc.value("license", std::string{});
    out.common.public_key_path = jc.value("public_key_path", std::string{});

    auto jt = j["tokenizer"];
    if (!jt.is_null()) {
        out.tokenizer.type = jt.value("type", std::string{"char"});
        out.tokenizer.vocab_path = jt.value("vocab_path", std::string{});
        out.tokenizer.merges_path = jt.value("merges_path", std::string{});
    }

    auto jm = j["model"];
    out.model.vocab_size = jm.value("vocab_size", out.model.vocab_size);
    out.model.n_layers = jm.value("n_layers", out.model.n_layers);
    out.model.n_heads = jm.value("n_heads", out.model.n_heads);
    out.model.d_model = jm.value("d_model", out.model.d_model);
    out.model.d_ff = jm.value("d_ff", out.model.d_ff);
    out.model.max_seq_len = jm.value("max_seq_len", out.model.max_seq_len);
    out.model.weights_type = jm.value("weights_type", std::string{"random"});
    out.model.weights_path = jm.value("weights_path", std::string{});

    return true;
}

} // namespace utils
