#include <engine/engine.hpp>
#include <api/api_server.hpp>
#include <utils/utils.hpp>
#include <model/transformer.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

namespace api {
    engine::Engine *g_engine_ptr = nullptr; // 전역 엔진 포인터
}

namespace {
// 간단한 CLI 옵션 구조체
struct CliOptions {
    std::string config = "config/engine-config.json";   // 기본 설정 경로
    std::string tokens_csv;                             // "1,2,3,4" 형태 (선택)
    bool show_help = false;                             // 도움말 표시
    bool serve = false;                                 // 서버 유지 실행
};

// 도움말 출력
void print_help() {
    std::cout << "Usage: ./mini_transformer [--config <path>] [--tokens \"1,2,3,4\"] [--serve] [--help|-h]\n"
              << "\n옵션 설명\n"
              << "  --config <path>   설정 파일 경로(기본: config/engine-config.json)\n"
              << "  --tokens <csv>    데모용 토큰 ID CSV (예: \"1,2,3,4\")\n"
              << "  --serve           HTTP 서버를 유지 실행(Ctrl+C로 종료)\n"
              << "  --help, -h        도움말 출력\n"
              << std::endl;
}

// 안전한 정수 파싱(범위 검사 포함)
bool safeParseInt(const std::string &s, int min_v, int max_v, int &out) {
    if (s.empty()) return false;
    try {
        long long v = std::stoll(s);
        if (v < min_v || v > max_v) return false;
        out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

// CSV("1,2,3") -> vector<int>, 개수/범위 제한
std::vector<int> parseTokensCsv(const std::string &csv, size_t max_count, int min_id, int max_id, bool &ok) {
    std::vector<int> out;
    ok = true;
    if (csv.size() > 4096) { // 비정상적으로 긴 입력 방지
        ok = false; return out;
    }
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // 공백 제거
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch){ return !std::isspace(ch); }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), item.end());
        int v;
        if (!safeParseInt(item, min_id, max_id, v)) { ok = false; break; }
        out.push_back(v);
        if (out.size() >= max_count) break; // 과도한 길이 방지
    }
    return out;
}

// 인자 파싱
CliOptions parse_args(int argc, char **argv) {
    CliOptions opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            opt.show_help = true;
        } else if (a == "--serve") {
            opt.serve = true;
        } else if (a == "--config" && i + 1 < argc) {
            std::string p = argv[++i];
            if (p.size() > 4096) {
                utils::logError("config 경로가 너무 깁니다.");
            } else {
                opt.config = p;
            }
        } else if (a == "--tokens" && i + 1 < argc) {
            opt.tokens_csv = argv[++i];
        } else {
            utils::logError("알 수 없는 인자: %s", a.c_str());
        }
    }
    return opt;
}
} // namespace

int main(int argc, char **argv) {
    // CLI 파싱 및 도움말
    CliOptions cli = parse_args(argc, argv);
    if (cli.show_help) {
        print_help();
        return 0;
    }

    // 프로그램 시작 로그
    utils::logInfo("mini-transformer 시작...");

    // 1) 엔진 구성(설정 파일 로드 -> 초기화)
    engine::Engine eng;
    const std::string cfg = cli.config; // 사용자 지정 또는 기본값
    if (eng.loadConfig(cfg) != engine::EngineState::Success) {
        utils::logError("설정 파일을 불러오지 못했습니다: %s", cfg.c_str());
        return static_cast<int>(engine::EngineState::EngineConfigLoadFailed);
    }
    if (eng.init() != engine::EngineState::Success) {
        utils::logError("엔진 초기화에 실패했습니다.");
        return static_cast<int>(engine::EngineState::EngineInitFailed);
    }

    api::g_engine_ptr = &eng; // 전역 엔진 포인터 설정

    utils::logInfo("엔진 초기화 완료. 설정 경로: %s", cfg.c_str());
    
    // --serve 모드: 서버를 유지 실행
    if (cli.serve) {
        utils::logInfo("--serve 모드: 서버를 유지합니다(Ctrl+C로 종료)...");
        const auto st = eng.run();
        eng.release();
        return static_cast<int>(st);
    }

    // 2) 간단한 미니 모델 데모(독립 실행) — 구조 확인용
    //    입력 토큰 CSV가 있으면 안전 파싱하여 사용, 없으면 기본 {1,2,3,4}
    const int demo_vocab = 3200;          // 데모용 vocab 상한
    const int demo_dmodel = 128;
    const int demo_nlayers = 1;
    const int demo_nheads = 2;
    const int demo_dff = 512;
    const int demo_maxseq = 64;           // 과도한 길이 방지

    std::vector<int> tokens;
    if (!cli.tokens_csv.empty()) {
        bool ok = true;
        tokens = parseTokensCsv(cli.tokens_csv, /*max_count*/ static_cast<size_t>(demo_maxseq), /*min_id*/ 0, /*max_id*/ demo_vocab - 1, ok);
        if (!ok || tokens.empty()) {
            utils::logError("--tokens 파싱 실패 또는 유효하지 않은 값. 기본 토큰을 사용합니다.");
            tokens = {1,2,3,4};
        }
    } else {
        tokens = {1,2,3,4};
    }

    mt::Transformer tf(/*vocab*/ demo_vocab, /*d_model*/ demo_dmodel, /*n_layers*/ demo_nlayers, /*n_heads*/ demo_nheads, /*d_ff*/ demo_dff, /*max_seq*/ demo_maxseq);
    int next = tf.next_token_argmax(tokens);
    std::cout << "next_token_argmax: " << next << std::endl;

    // 3) 리소스 정리(서버 등)
    eng.release();
    utils::logInfo("mini-transformer 종료.");
    return 0;
}
