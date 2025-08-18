#include <engine/engine.hpp>
#include <api/api_server.hpp>
#include <utils/utils.hpp>
#include <secure/secure.hpp>
// nlohmann::json 전체 정의를 포함합니다(전방 선언으로 인한 모호성 제거)
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <csignal>
#include <chrono>
#include <thread>
#include <sstream>
#include <algorithm>

// ─────────────────────────────────────────────────────────────
// 내부(익명) 네임스페이스: 신호 처리기와 전역 플래그
// ─────────────────────────────────────────────────────────────
namespace {
std::atomic<bool> g_should_exit{false};

void signalHandler(int /*signum*/) {
    // Ctrl+C 등으로 종료 요청이 들어오면 루프를 빠져나옵니다.
    g_should_exit = true; 
}
}

namespace engine {

Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::createSharedEngine() { return std::make_unique<Engine>(); }

// JSON에서 행렬/벡터를 읽어오는 보조 함수들(간단한 스키마)
static bool load_matrix(const nlohmann::json& j, const char* key, int rows, int cols, mt::Tensor2D& dst) {
    if (!j.contains(key)) return false;
    const auto& M = j.at(key);
    if (!M.is_array() || (int)M.size() != rows) return false;
    dst = mt::Tensor2D(rows, cols);
    for (int r=0; r<rows; ++r) {
        const auto& row = M.at(r);
        if (!row.is_array() || (int)row.size() != cols) return false;
        for (int c=0; c<cols; ++c) dst(r,c) = static_cast<float>(row.at(c).get<double>());
    }
    return true;
}

static bool load_vector(const nlohmann::json& j, const char* key, int len, std::vector<float>& dst) {
    if (!j.contains(key)) return false;
    const auto& V = j.at(key);
    if (!V.is_array() || (int)V.size() != len) return false;
    dst.resize(len);
    for (int i=0; i<len; ++i) dst[i] = static_cast<float>(V.at(i).get<double>());
    return true;
}

static bool populate_from_json(const nlohmann::json& root, mt::Transformer& m) {
    bool any=false;
    // 임베딩/출력 투영
    any |= load_matrix(root, "tok_emb", m.vocab, m.d_model, m.tok_emb);
    any |= load_matrix(root, "pos_emb", m.max_seq, m.d_model, m.pos_emb);
    any |= load_matrix(root, "Wout",    m.d_model, m.vocab, m.Wout);
    if (root.contains("ln_f")) {
        const auto& ln = root.at("ln_f");
        any |= load_vector(ln, "gamma", m.d_model, m.ln_f.gamma);
        any |= load_vector(ln, "beta",  m.d_model, m.ln_f.beta);
    }
    // 블록별 파라미터
    if (root.contains("blocks")) {
        const auto& bs = root.at("blocks");
        int L = std::min<int>((int)bs.size(), m.n_layers);
        for (int i=0; i<L; ++i) {
            const auto& bj = bs.at(i);
            if (bj.contains("ln1")) {
                const auto& ln = bj.at("ln1");
                any |= load_vector(ln, "gamma", m.d_model, m.blocks[i].ln1.gamma);
                any |= load_vector(ln, "beta",  m.d_model, m.blocks[i].ln1.beta);
            }
            if (bj.contains("ln2")) {
                const auto& ln = bj.at("ln2");
                any |= load_vector(ln, "gamma", m.d_model, m.blocks[i].ln2.gamma);
                any |= load_vector(ln, "beta",  m.d_model, m.blocks[i].ln2.beta);
            }
            if (bj.contains("mha")) {
                const auto& mha = bj.at("mha");
                any |= load_matrix(mha, "Wq", m.d_model, m.d_model, m.blocks[i].mha.Wq);
                any |= load_matrix(mha, "Wk", m.d_model, m.d_model, m.blocks[i].mha.Wk);
                any |= load_matrix(mha, "Wv", m.d_model, m.d_model, m.blocks[i].mha.Wv);
                any |= load_matrix(mha, "Wo", m.d_model, m.d_model, m.blocks[i].mha.Wo);
            }
            if (bj.contains("ffn")) {
                const auto& ffn = bj.at("ffn");
                any |= load_matrix(ffn, "W1", m.d_model, m.d_ff,  m.blocks[i].ffn.W1);
                any |= load_matrix(ffn, "W2", m.d_ff,   m.d_model, m.blocks[i].ffn.W2);
            }
        }
    }
    return any;
}

EngineState Engine::loadConfig(const std::string &filepath) {
    m_config_filepath = filepath; // 이후 상대 경로 보정에 사용
    if (!utils::loadEngineConfig(filepath, m_config)) {
        utils::logError("Failed to load engine config: %s", filepath.c_str());
        return EngineState::EngineConfigLoadFailed;
    }
    return EngineState::Success;
}

EngineState Engine::init() {
    // 1) 라이선스/보안(스텁)
    std::string license_json;
    std::string license_json_path = m_config.common.license;
    std::string public_key_path = m_config.common.public_key_path;

    if (!public_key_path.empty()) {
        std::filesystem::path config_dir = std::filesystem::path(m_config_filepath).parent_path();
        std::filesystem::path abs_public_key_path = config_dir / public_key_path;
        public_key_path = abs_public_key_path.string();
        utils::logInfo("PUBLIC KEY PATH (from config): %s", public_key_path.c_str());
    }

    utils::logInfo("LICENSE FILE PATH: %s", license_json_path.c_str());

    if (secure::loadLicenseFile(license_json_path, license_json)) {
        try {
            auto root = nlohmann::json::parse(license_json);
            if (root.contains("public_key_path") && !root["public_key_path"].is_null()) {
                std::string license_public_key_path = root["public_key_path"].get<std::string>();
                std::filesystem::path license_dir = std::filesystem::path(license_json_path).parent_path();
                std::filesystem::path abs_license_public_key_path = license_dir / license_public_key_path;
                public_key_path = abs_license_public_key_path.string();
                utils::logInfo("PUBLIC KEY PATH (from license): %s", public_key_path.c_str());
            }
        } catch (const std::exception& e) {
            utils::logError("Failed to parse license file: %s", e.what());
        }
    }

    utils::logInfo("PUBLIC KEY PATH: %s", public_key_path.c_str());

    if (!secure::AntiPiracy::verifyProgramIntegrity()) {
        utils::logError("Program integrity verification failed.");
        return EngineState::EngineInitFailed;
    }

    if (!secure::AntiPiracy::activateOnlineFromJson(license_json)) {
        utils::logError("Online activation failed.");
        return EngineState::EngineInitFailed;
    }

    if (!secure::SignatureVerifier::verifySignatureFromJson(license_json, license_json_path)) {
        utils::logError("Signature verification failed.");
        return EngineState::EngineInitFailed;
    }

    // 2) 모델 구성(미니 트랜스포머)
    m_model = std::make_unique<mt::Transformer>(
        m_config.model.vocab_size,
        m_config.model.d_model,
        m_config.model.n_layers,
        m_config.model.n_heads,
        m_config.model.d_ff,
        m_config.model.max_seq_len
    );

    // 3) (선택) 외부 JSON 가중치 로드
    if (m_config.model.weights_type == "json" && !m_config.model.weights_path.empty()) {
        std::filesystem::path config_dir = std::filesystem::path(m_config_filepath).parent_path();
        std::filesystem::path weights_path = config_dir / m_config.model.weights_path;
        utils::logInfo("Loading weights JSON: %s", weights_path.string().c_str());
        try {
            std::ifstream ifs(weights_path);
            if (!ifs) {
                utils::logError("Failed to open weights file");
            } else {
                nlohmann::json jweights; ifs >> jweights;
                if (populate_from_json(jweights, *m_model)) {
                    utils::logInfo("Weights populated successfully");
                } else {
                    utils::logError("Weights JSON missing or mismatched shapes; using random-initialized parts");
                }
            }
        } catch (const std::exception& e) {
            utils::logError("Weights load error: %s", e.what());
        }
    } else {
        utils::logInfo("Using random-initialized weights");
    }

    // 4) API 서버 초기화(HTTP)
    try {
        std::filesystem::path config_dir = std::filesystem::path(m_config_filepath).parent_path();
        std::filesystem::path doc_root = (config_dir / ".." / "web").lexically_normal();
        utils::logInfo("API Server doc root: %s", doc_root.string().c_str());
        utils::logInfo("API Server initializing at 0.0.0.0:%d", m_config.common.api_port);
        m_api_server = std::make_unique<api::ApiServer>("0.0.0.0", m_config.common.api_port, doc_root.string());
        m_api_server->init();

        // 핸들러
        m_api_server->setPredictHandler([this](const std::vector<int>& tokens){
            if (!m_model) return std::string("<pre>Model not initialized</pre>");
            if (tokens.empty()) return std::string("<pre>토큰을 입력하세요 (예: 1,2,3,4)</pre>");
            auto logits = m_model->forward(tokens);
            int T = logits.rows; int V = logits.cols; int last = T-1;
            // top-5 추출
            std::vector<std::pair<int,float>> items; items.reserve(V);
            for (int v=0; v<V; ++v) items.emplace_back(v, logits(last,v));
            std::partial_sort(items.begin(), items.begin()+std::min(5,(int)items.size()), items.end(),
                              [](auto& a, auto& b){ return a.second > b.second; });
            std::ostringstream oss;
            oss << "<div>입력 토큰: [";
            for (size_t i=0;i<tokens.size();++i){ if (i) oss << ", "; oss << tokens[i]; }
            oss << "]</div>";
            oss << "<div>로짓 행렬 크기: (" << T << " x " << V << ")</div>";
            oss << "<h4>Top-5 (마지막 토큰 기준)</h4>";
            oss << "<table><thead><tr><th>토큰ID</th><th>logit</th></tr></thead><tbody>";
            int topk = std::min(5,(int)items.size());
            for (int i=0;i<topk;++i){ oss << "<tr><td>" << items[i].first << "</td><td>" << items[i].second << "</td></tr>"; }
            oss << "</tbody></table>";
            int next = m_model->next_token_argmax(tokens);
            oss << "<div><b>다음 토큰(argmax): </b>" << next << "</div>";
            return oss.str();
        });

        utils::logInfo("API Server initialized successfully.");
    } catch (const std::exception& e) {
        utils::logError("API Server init failed: %s", e.what());
        return EngineState::EngineInitFailed;
    }

    // 5) 간단한 데모 순전파(토큰 4개)
    std::vector<int> demo = {1,2,3,4};
    auto logits = m_model->forward(demo);
    utils::logInfo("Model demo forward done: logits shape (%d x %d)", logits.rows, logits.cols);

    utils::logInfo("Engine initialization complete.");
    return EngineState::Success;
}

EngineState Engine::run() const {
    // 메인 루프: SIGINT 수신 시 종료
    std::signal(SIGINT, signalHandler);
    utils::logInfo("Engine is now running. Press Ctrl+C to terminate...");

    std::thread api_thread([this]() { m_api_server->start(); });

    while (!g_should_exit) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    utils::logInfo("Gracefully exiting engine loop. Performing cleanup...");

    if (m_api_server) { m_api_server->stop(); }
    if (api_thread.joinable()) { api_thread.join(); }

    return EngineState::Success;
}

void Engine::updateAll() { updateDisplay(); updateDeviceInfo(); updateEventArea(); }
void Engine::updateDeviceInfo() { }
void Engine::updateEventArea() { }
void Engine::updateDisplay() { }

EngineState Engine::release() {
    m_api_server.reset();
    return EngineState::Success;
}

} // namespace engine
