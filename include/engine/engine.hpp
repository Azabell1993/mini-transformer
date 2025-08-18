#pragma once
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <model/transformer.hpp>

namespace api { class ApiServer; }

namespace engine {

// 엔진 동작 상태를 나타내는 열거형입니다.
// - Success: 정상 동작
// - EngineConfigLoadFailed: 설정 파일(JSON) 로드 실패
// - EngineInitFailed: 초기화(보안/서버/모델) 과정 실패
// - EngineLicenseCheckFailed: 라이선스 검증 실패
enum class EngineState : int {
    Success = 0,
    EngineConfigLoadFailed = 10,
    EngineInitFailed = 11,
    EngineLicenseCheckFailed = 12,
};

// 공통 실행 설정(네트워크 포트, 라이선스 파일 경로 등)을 담는 구조체입니다.
struct CommonConfig {
    int api_port = 18080;          // API 서버 포트 번호
    std::string license;           // 라이선스 JSON 경로(스텁)
    std::string public_key_path;   // 공개키 경로(스텁)
};

// 토크나이저 설정: BPE 또는 문자 단위(char) 등
// - 본 미니 엔진은 토크나이저 자체를 구현하지 않았으며, 경로만 보관합니다.
struct TokenizerConfig {
    std::string type;        // "bpe" | "char"
    std::string vocab_path;  // BPE vocab.json
    std::string merges_path; // BPE merges.txt
};

// 모델 구조 하이퍼파라미터와 가중치 로딩 방법을 정의합니다.
// - vocab_size, d_model 등의 숫자는 네트워크 크기를 결정합니다.
// - weights_type: "random"(무작위 초기화) 또는 "json"(외부 JSON 가중치 로드)
// - weights_path: weights_type=="json"일 때 사용할 파일 경로
struct ModelConfig {
    int vocab_size = 32000;
    int n_layers = 1;
    int n_heads = 2;
    int d_model = 128;
    int d_ff = 512;
    int max_seq_len = 64;
    std::string weights_type;   // "random" | "json"
    std::string weights_path;   // weights_type=="json"일 때 사용
};

// 엔진 전체 설정(공통 + 토크나이저 + 모델)
struct EngineConfig {
    CommonConfig common;
    TokenizerConfig tokenizer;
    ModelConfig model;
};

// Engine 클래스는 다음을 담당합니다.
// - 설정 파일 로드(loadConfig)
// - 초기화(init): 보안 스텁, 모델 생성/가중치 로드, API 서버 초기화
// - 실행(run): 서버 스레드 구동 및 루프 유지
// - 해제(release): 리소스 정리
class Engine {
public:
    ~Engine();
    std::unique_ptr<mt::Transformer> m_model; // 미니 트랜스포머 모델 포인터
    static std::unique_ptr<Engine> createSharedEngine();

    EngineState loadConfig(const std::string &filepath);
    EngineState init();
    EngineState run() const;
    EngineState release();

    // 예시용 업데이트 함수(현재는 비어있음)
    void updateAll();
    void updateDeviceInfo();
    void updateEventArea();
    void updateDisplay();

private:
    std::unique_ptr<api::ApiServer> m_api_server; // 간단한 HTTP 서버
    EngineConfig m_config;                        // 런타임 설정 값
    std::string m_config_filepath;                // 설정 파일 실제 경로(상대->절대 변환에 사용)
    // std::unique_ptr<mt::Transformer> m_model;     // 미니 트랜스포머 네트워크(순전파용)
};

} // namespace engine
