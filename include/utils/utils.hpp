#pragma once
#include <string>
#include <engine/engine.hpp>

namespace utils {

// 엔진 설정(JSON)을 읽어 EngineConfig 구조체에 채워 넣습니다.
bool loadEngineConfig(const std::string& path, engine::EngineConfig& out);

// 간단한 로그 유틸리티(표준 에러로 출력)
void logInfo(const char* fmt, ...);
void logError(const char* fmt, ...);

}
