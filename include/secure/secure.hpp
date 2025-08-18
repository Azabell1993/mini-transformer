#pragma once
#include <string>

namespace secure {

// 라이선스/보안 관련 스텁 함수 선언부입니다.
// 실제 서비스 적용 시, 아래 함수들을 실제 검증 로직으로 교체해야 합니다.

// 라이선스 JSON 파일을 읽어 문자열로 반환합니다.
bool loadLicenseFile(const std::string& path, std::string& out);

struct AntiPiracy {
    // 프로그램 무결성 검증(스텁: 항상 true 반환)
    static bool verifyProgramIntegrity();
    // 온라인 활성화 처리(스텁: 항상 true 반환)
    static bool activateOnlineFromJson(const std::string& json);
};

struct SignatureVerifier {
    // 라이선스 서명 검증(스텁: 항상 true 반환)
    static bool verifySignatureFromJson(const std::string& json, const std::string& licensePath);
};

} // namespace secure
