#include <api/api_server.hpp>
#include <utils/utils.hpp>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/core/file.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <nlohmann/json.hpp>

namespace net = boost::asio;
namespace http = boost::beast::http;

namespace api {

extern engine::Engine* g_engine_ptr;

ApiServer::ApiServer(const std::string& host, int port, const std::string& docRoot)
    : m_host(host), m_port(port), m_doc_root(docRoot) {}
ApiServer::~ApiServer() { stop(); }

void ApiServer::init() { /* 현재는 준비할 리소스 없음 */ }

static std::string url_decode(const std::string& in) {
    std::string out; out.reserve(in.size());
    for (size_t i=0; i<in.size(); ++i) {
        if (in[i]=='+') { out.push_back(' '); }
        else if (in[i]=='%' && i+2<in.size()) {
            std::string hex = in.substr(i+1,2);
            char ch = static_cast<char>(std::strtol(hex.c_str(), nullptr, 16));
            out.push_back(ch); i+=2;
        } else { out.push_back(in[i]); }
    }
    return out;
}

static std::vector<int> parse_tokens_from_form(const std::string& body) {
    // form-urlencoded: tokens=1%2C2%2C3
    std::string key = "tokens=";
    size_t pos = body.find(key);
    if (pos == std::string::npos) return {};
    std::string enc = body.substr(pos + key.size());
    std::string s = url_decode(enc);
    std::vector<int> out; std::stringstream ss(s); std::string item;
    while (std::getline(ss, item, ',')) {
        try { if (!item.empty()) out.push_back(std::stoi(item)); } catch(...) {}
    }
    return out;
}

static void write_text_response(net::ip::tcp::socket& socket, unsigned version, http::status status, const std::string& body, const char* content_type) {
    http::response<http::string_body> res{status, version};
    res.set(http::field::server, "mini-transformer");
    res.set(http::field::content_type, content_type);
    res.body() = body;
    res.prepare_payload();
    http::write(socket, res);
}

static std::string read_file_to_string(const std::filesystem::path& p) {
    utils::logInfo("Attempting to read file: %s", p.string().c_str());
    
    // 파일 존재 확인
    if (!std::filesystem::exists(p)) {
        utils::logError("File does not exist: %s", p.string().c_str());
        return {};
    }
    
    // 절대 경로 로깅
    auto abs_path = std::filesystem::absolute(p);
    utils::logInfo("Absolute path: %s", abs_path.string().c_str());
    
    std::ifstream ifs(p, std::ios::in | std::ios::binary);
    if (!ifs) {
        utils::logError("Failed to open file: %s", p.string().c_str());
        return {};
    }
    
    std::ostringstream oss; 
    oss << ifs.rdbuf();
    std::string content = oss.str();
    utils::logInfo("Successfully read file: %s (%zu bytes)", p.string().c_str(), content.size());
    return content;
}

// 단일 요청을 처리하는 세션 함수(간단 라우팅)
static void handle_session(net::ip::tcp::socket socket, const std::string docRoot, ApiServer::PredictHandler predict) {
    try {
        boost::beast::flat_buffer buffer;
        http::request<http::string_body> req;
        http::read(socket, buffer, req);

        std::string target = std::string(req.target());
        
        // 라우팅
        if (req.method()==http::verb::get && (target=="/" || target=="/index.html")) {
            if (!docRoot.empty()) {
                auto p = std::filesystem::path(docRoot) / "index.html";
                auto body = read_file_to_string(p);
                if (!body.empty()) {
                    write_text_response(socket, req.version(), http::status::ok, body, "text/html; charset=utf-8");
                } else {
                    write_text_response(socket, req.version(), http::status::not_found, "index.html not found", "text/plain; charset=utf-8");
                }
            } else {
                write_text_response(socket, req.version(), http::status::ok, "<html><body><h1>mini-transformer</h1></body></html>", "text/html; charset=utf-8");
            }
        }
        else if (req.method()==http::verb::post && target=="/set_config") {
            nlohmann::json j = {};
            try { j = nlohmann::json::parse(req.body()); } catch (...) {}
            bool ok = false;
            int vocab = 0, d_model = 0, n_layers = 0, n_heads = 0, d_ff = 0, max_seq = 0;
            if (j.is_object()) {
            vocab = j.value("vocab_size", 0);
            d_model = j.value("d_model", 0);
            n_layers = j.value("n_layers", 0);
            n_heads = j.value("n_heads", 0);
            d_ff = j.value("d_ff", 0);
            max_seq = j.value("max_seq_len", 0);

            utils::logInfo("set_config 요청: vocab=%d, d_model=%d, n_layers=%d, n_heads=%d, d_ff=%d, max_seq_len=%d",
                vocab, d_model, n_layers, n_heads, d_ff, max_seq);

            if (vocab >= 100 && vocab <= 10000 &&
                d_model > 0 && n_layers > 0 && n_heads > 0 &&
                d_ff > 0 && max_seq > 0 &&
                d_model % n_heads == 0) {
                extern engine::Engine* g_engine_ptr;
                if (g_engine_ptr) {
                g_engine_ptr->m_model = std::make_unique<mt::Transformer>(vocab, d_model, n_layers, n_heads, d_ff, max_seq);
                ok = true;
                } else {
                utils::logError("g_engine_ptr is null");
                }
            } else {
                utils::logError("set_config 값 검증 실패: vocab=%d, d_model=%d, n_layers=%d, n_heads=%d, d_ff=%d, max_seq_len=%d",
                vocab, d_model, n_layers, n_heads, d_ff, max_seq);
            }
            } else {
            utils::logError("set_config 요청 JSON 형식 오류");
            }
            std::string res = ok ? R"({"success":true})" : R"({"success":false,"message":"설정값 오류"})";
            write_text_response(socket, req.version(), http::status::ok, res, "application/json; charset=utf-8");
        }
        else if (req.method()==http::verb::get && target=="/style.css") {
            if (!docRoot.empty()) {
                auto p = std::filesystem::path(docRoot) / "style.css";
                auto body = read_file_to_string(p);
                if (!body.empty()) {
                    write_text_response(socket, req.version(), http::status::ok, body, "text/css; charset=utf-8");
                } else {
                    write_text_response(socket, req.version(), http::status::not_found, "style.css not found", "text/plain; charset=utf-8");
                }
            } else {
                write_text_response(socket, req.version(), http::status::not_found, "style.css not found", "text/plain; charset=utf-8");
            }
        } else if (req.method()==http::verb::get && target=="/main.js") {
            if (!docRoot.empty()) {
                auto p = std::filesystem::path(docRoot) / "main.js";
                auto body = read_file_to_string(p);
                if (!body.empty()) {
                    write_text_response(socket, req.version(), http::status::ok, body, "application/javascript; charset=utf-8");
                } else {
                    write_text_response(socket, req.version(), http::status::not_found, "main.js not found", "text/plain; charset=utf-8");
                }
            } else {
                write_text_response(socket, req.version(), http::status::not_found, "main.js not found", "text/plain; charset=utf-8");
            }
        } else if (req.method()==http::verb::post && target=="/predict") {
            std::vector<int> tokens;
            // 간단히 application/x-www-form-urlencoded 가정
            if (req[http::field::content_type].find("application/x-www-form-urlencoded") != std::string::npos) {
                tokens = parse_tokens_from_form(req.body());
            }
            std::string html;
            if (predict) {
                html = predict(tokens);
            } else {
                html = "<pre>No predict handler set</pre>";
            }
            write_text_response(socket, req.version(), http::status::ok, html, "text/html; charset=utf-8");
        } else {
            write_text_response(socket, req.version(), http::status::not_found, "Not Found", "text/plain; charset=utf-8");
        }

        socket.shutdown(net::ip::tcp::socket::shutdown_send);
    } catch (...) {
        // 단순 예제이므로 에러는 무시
    }
}

void ApiServer::start() {
    if (m_running.exchange(true)) return; // 이미 실행 중이면 무시
    m_thread = std::thread([this]() {
        try {
            net::io_context ioc{1};
            
            // macOS 호환성을 위해 주소 바인딩 방식 개선
            net::ip::tcp::endpoint endpoint;
            if (m_host == "0.0.0.0") {
                // 모든 IPv4 인터페이스에 바인딩
                endpoint = net::ip::tcp::endpoint(net::ip::tcp::v4(), static_cast<unsigned short>(m_port));
            } else {
                // 특정 주소에 바인딩
                endpoint = net::ip::tcp::endpoint(net::ip::make_address(m_host), static_cast<unsigned short>(m_port));
            }
            
            utils::logInfo("Creating acceptor for endpoint %s:%d", endpoint.address().to_string().c_str(), endpoint.port());
            
            net::ip::tcp::acceptor acceptor{ioc};
            
            boost::system::error_code ec;
            acceptor.open(endpoint.protocol(), ec);
            if (ec) {
                utils::logError("Failed to open acceptor: %s", ec.message().c_str());
                return;
            }
            
            acceptor.set_option(net::ip::tcp::acceptor::reuse_address(true), ec);
            if (ec) {
                utils::logError("Failed to set reuse_address: %s", ec.message().c_str());
                return;
            }
            
            // macOS에서 SO_REUSEPORT 설정 시도
            #ifdef SO_REUSEPORT
            acceptor.set_option(boost::asio::socket_base::reuse_address(true), ec);
            #endif
            
            acceptor.bind(endpoint, ec);
            if (ec) {
                utils::logError("Failed to bind to endpoint: %s", ec.message().c_str());
                return;
            }
            
            acceptor.listen(net::socket_base::max_listen_connections, ec);
            if (ec) {
                utils::logError("Failed to listen: %s", ec.message().c_str());
                return;
            }
            
            utils::logInfo("Server successfully bound and listening on %s:%d", endpoint.address().to_string().c_str(), endpoint.port());
            
            while (m_running.load()) {
                net::ip::tcp::socket socket{ioc};
                
                // 논블로킹 accept 시도
                acceptor.async_accept(socket, [this, &socket](boost::system::error_code ec) {
                    if (!ec && m_running.load()) {
                        std::thread(&handle_session, std::move(socket), m_doc_root, m_predict).detach();
                    } else if (ec && m_running.load()) {
                        utils::logError("Async accept failed: %s", ec.message().c_str());
                    }
                });
                
                // 짧은 시간 동안 이벤트 처리
                ioc.run_for(std::chrono::milliseconds(100));
                ioc.restart();
            }
        } catch (const std::exception& e) {
            utils::logError("ApiServer exception: %s", e.what());
        }
    });
}

void ApiServer::stop() {
    if (!m_running.exchange(false)) return; // 이미 멈춰있음
    if (m_thread.joinable()) m_thread.join();
}

} // namespace api
