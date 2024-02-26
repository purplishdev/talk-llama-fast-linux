// Talk with AI
//

#include "common-sdl.h"
#include "common.h"
#include "console.h"
#include "console.cpp"
#include "whisper.h"
#include "llama.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <sstream>


#include <algorithm> 
#include <cctype>
#include <locale>
#include <codecvt>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* #ifdef _WIN32 */
/* #include <winsock2.h> */
/* #include <windows.h> */
/* #endif */
/* #include <sys/types.h> */
/* #pragma comment(lib,"ws2_32.lib") */



#include <clocale>
#include <curl/curl.h>
#include <unordered_set>
#include <ctype.h>
#include <map>
#include <iterator>
#include <ctime>

#include <cstring>
#include <iostream>
#include <unistd.h>
#include <netdb.h>

std::string send_http_request(const std::string& hostname, const std::string& port, const std::string& path, const std::string& request_body) {
    struct addrinfo hints{}, *server_info, *p;
    int sockfd;
    std::string response;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(hostname.c_str(), port.c_str(), &hints, &server_info) != 0) {
        perror("getaddrinfo failed");
        return "";
    }

    for(p = server_info; p != nullptr; p = p->ai_next) {
        if ((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
            perror("client: socket");
            continue;
        }

        if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sockfd);
            perror("client: connect");
            continue;
        }

        break; // successfully connected
    }

    if (p == nullptr) {
        fprintf(stderr, "client: failed to connect\n");
        return "";
    }

    freeaddrinfo(server_info);

    std::string http_request = "POST " + path + " HTTP/1.1\r\n";
    http_request += "Host: " + hostname + "\r\n";
    http_request += "Content-Length: " + std::to_string(request_body.length()) + "\r\n";
    http_request += "Content-Type: application/x-www-form-urlencoded\r\n\r\n";
    http_request += request_body;

    if (send(sockfd, http_request.c_str(), http_request.length(), 0) == -1) {
        perror("send");
        close(sockfd);
        return "";
    }

    // Receive response
    const int buf_size = 4096;
    char buf[buf_size];
    ssize_t num_bytes;
    while ((num_bytes = recv(sockfd, buf, buf_size - 1, 0)) > 0) {
        buf[num_bytes] = '\0';
        response += std::string(buf);
    }

    if (num_bytes == -1) {
        perror("recv");
    }

    close(sockfd);

    return response;
}
std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    auto * model = llama_get_model(ctx);

    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t voice_ms   = 10000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t n_gpu_layers = 999;

    float vad_thold  = 0.6f;
    float vad_last_ms  = 1250;
    float freq_thold = 100.0f;

    bool speed_up       = false;
    bool translate      = false;
    bool print_special  = false;
    bool print_energy   = false;
    bool no_timestamps  = true;
    bool verbose_prompt = false;
    bool use_gpu        = true;

    std::string person      = "Georgi";
    std::string bot_name    = "LLaMA";
    std::string xtts_voice  = "emma_1";
    std::string wake_cmd    = "";
    std::string heard_ok    = "";
    std::string language    = "en";
    std::string model_wsp   = "models/ggml-base.en.bin";
    std::string model_llama = "models/ggml-llama-7B.bin";
    std::string speak       = "./examples/talk-llama/speak";
	std::string xtts_control_path = "/home/purpledev/talk-llama-fast/xtts/xtts_play_allowed.txt";
    std::string xtts_url = "http://localhost:8020/";
    std::string google_url = "http://localhost:8003/";
    std::string prompt      = "";
    std::string fname_out;
    std::string path_session = "";       // path to file for saving/loading model eval state
    int32_t ctx_size = 2048;      
    int32_t n_predict = 64;      
    float temp = 0.9;      
    float top_k = 40;      
    float top_p = 1.0f;      
    float repeat_penalty = 1.10;   
};

void whisper_print_usage(int argc, const char ** argv, const whisper_params & params);

bool whisper_params_parse(int argc, const char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")        { params.n_threads      = std::stoi(argv[++i]); }
        else if (arg == "-vms" || arg == "--voice-ms")       { params.voice_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")        { params.capture_id     = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")     { params.max_tokens     = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")      { params.audio_ctx      = std::stoi(argv[++i]); }
        else if (arg == "-ngl" || arg == "--n-gpu-layers")   { params.n_gpu_layers   = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")      { params.vad_thold      = std::stof(argv[++i]); }
        else if (arg == "-vlm" || arg == "--vad-last-ms")    { params.vad_last_ms    = std::stoi(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")     { params.freq_thold     = std::stof(argv[++i]); }
        else if (arg == "-su"  || arg == "--speed-up")       { params.speed_up       = true; }
        else if (arg == "-tr"  || arg == "--translate")      { params.translate      = true; }
        else if (arg == "-ps"  || arg == "--print-special")  { params.print_special  = true; }
        else if (arg == "-pe"  || arg == "--print-energy")   { params.print_energy   = true; }
        else if (arg == "-vp"  || arg == "--verbose-prompt") { params.verbose_prompt = true; }
        else if (arg == "-ng"  || arg == "--no-gpu")         { params.use_gpu        = false; }
        else if (arg == "-p"   || arg == "--person")         { params.person         = argv[++i]; }
        else if (arg == "-bn"   || arg == "--bot-name")      { params.bot_name       = argv[++i]; }
        else if (arg == "--session")                         { params.path_session   = argv[++i]; }
        else if (arg == "-w"   || arg == "--wake-command")   { params.wake_cmd       = argv[++i]; }
        else if (arg == "-ho"  || arg == "--heard-ok")       { params.heard_ok       = argv[++i]; }
        else if (arg == "-l"   || arg == "--language")       { params.language       = argv[++i]; }
        else if (arg == "-mw"  || arg == "--model-whisper")  { params.model_wsp      = argv[++i]; }
        else if (arg == "-ml"  || arg == "--model-llama")    { params.model_llama    = argv[++i]; }
        else if (arg == "-s"   || arg == "--speak")          { params.speak          = argv[++i]; }
        else if (arg == "--ctx_size")                        { params.ctx_size       = std::stoi(argv[++i]); }
        else if (arg == "-n"   || arg == "--n_predict")      { params.n_predict      = std::stoi(argv[++i]); }
        else if (arg == "--temp")     						 { params.temp           = std::stof(argv[++i]); }
        else if (arg == "--top_k")     						 { params.top_k          = std::stof(argv[++i]); }
        else if (arg == "--top_p")     						 { params.top_p          = std::stof(argv[++i]); }
        else if (arg == "--repeat_penalty")     			 { params.repeat_penalty = std::stof(argv[++i]); }
        else if (arg == "--xtts-voice")                      { params.xtts_voice     = argv[++i]; }
        else if (arg == "--xtts-url")                        { params.xtts_url = argv[++i]; }
        else if (arg == "--google-url")                      { params.google_url = argv[++i]; }
        else if (arg == "--xtts-control-path")               { params.xtts_control_path = argv[++i]; }
        else if (arg == "--prompt-file")                     {
            std::ifstream file(argv[++i]);
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
        else if (arg == "-f"   || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-ng"  || arg == "--no-gpu")        { params.use_gpu       = false; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, const char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N     [%-7d] voice duration in milliseconds\n",              params.voice_ms);
    fprintf(stderr, "  -c ID,    --capture ID     [%-7d] capture device ID\n",                           params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N   [%-7d] maximum number of tokens per audio chunk\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N    [%-7d] audio context size (0 - all)\n",                params.audio_ctx);
    fprintf(stderr, "  -ngl N,   --n-gpu-layers N [%-7d] number of layers to store in VRAM\n",           params.n_gpu_layers);
    fprintf(stderr, "  -vth N,   --vad-thold N    [%-7.2f] voice activity detection threshold\n",        params.vad_thold);
    /* fprintf(stderr, "  -vlm N,   --vad-last-ms N  [%-7d] vad min silence after speech, ms\n",       	 params.vad_last_ms); */
	fprintf(stderr, "  -vlm N,   --vad-last-ms N  [%-7.0f] vad min silence after speech, ms\n", static_cast<double>(params.vad_last_ms));
    fprintf(stderr, "  -fth N,   --freq-thold N   [%-7.2f] high-pass frequency cutoff\n",                params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up       [%-7s] speed up audio by x2 (reduced accuracy)\n",     params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate      [%-7s] translate from source language to english\n",   params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special  [%-7s] print special tokens\n",                        params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,      --print-energy   [%-7s] print sound energy (for debugging)\n",          params.print_energy ? "true" : "false");
    fprintf(stderr, "  -vp,      --verbose-prompt [%-7s] print prompt at start\n",                       params.verbose_prompt ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu         [%-7s] disable GPU\n",                                 params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -p NAME,  --person NAME    [%-7s] person name (for prompt selection)\n",          params.person.c_str());
    fprintf(stderr, "  -bn NAME, --bot-name NAME  [%-7s] bot name (to display)\n",                       params.bot_name.c_str());
    fprintf(stderr, "  -w TEXT,  --wake-command T [%-7s] wake-up command to listen for\n",               params.wake_cmd.c_str());
    fprintf(stderr, "  -ho TEXT, --heard-ok TEXT  [%-7s] said by TTS before generating reply\n",         params.heard_ok.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG  [%-7s] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper  [%-7s] whisper model file\n",                          params.model_wsp.c_str());
    fprintf(stderr, "  -ml FILE, --model-llama    [%-7s] llama model file\n",                            params.model_llama.c_str());
    fprintf(stderr, "  -s FILE,  --speak TEXT     [%-7s] command for TTS\n",                             params.speak.c_str());
    fprintf(stderr, "  --prompt-file FNAME        [%-7s] file with custom prompt to start dialog\n",     "");
    fprintf(stderr, "  --session FNAME                   file to cache model state in (may be large!) (default: none)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME     [%-7s] text output file name\n",                       params.fname_out.c_str());
    fprintf(stderr, "   --ctx_size N              [%-7d] Size of the prompt context\n",                  params.ctx_size);
    fprintf(stderr, "  -n N, --n_predict N        [%-7d] Number of tokens to predict\n",                 params.n_predict);
    fprintf(stderr, "  --temp N                   [%-7.2f] Temperature \n",                              params.temp);
    fprintf(stderr, "  --top_k N                  [%-7.2f] top_k \n",                                    params.top_k);
    fprintf(stderr, "  --top_p N                  [%-7.2f] top_p \n",                                    params.top_p);
    fprintf(stderr, "  --repeat_penalty N         [%-7.2f] repeat_penalty \n",                           params.repeat_penalty);
	fprintf(stderr, "  --xtts-voice NAME          [%-7s] xtts voice without .wav\n",                     params.xtts_voice.c_str());
	fprintf(stderr, "  --xtts-url TEXT            [%-7s] xtts/silero server URL, with trailing slash\n", params.xtts_url.c_str());
	fprintf(stderr, "  --xtts-control-path FNAME  [%-7s] path to xtts_play_allowed.txt",                 params.xtts_control_path.c_str());
	fprintf(stderr, "  --google-url TEXT          [%-7s] langchain google-serper server URL, with /\n",  params.google_url.c_str());
    fprintf(stderr, "\n");
}

std::string transcribe(
        whisper_context * ctx,
        const whisper_params & params,
        const std::vector<float> & pcmf32,
        const std::string prompt_text,
        float & prob,
        int64_t & t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    prob = 0.0f;
    t_ms = 0;

    std::vector<whisper_token> prompt_tokens;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    prompt_tokens.resize(1024);
    prompt_tokens.resize(whisper_tokenize(ctx, prompt_text.c_str(), prompt_tokens.data(), prompt_tokens.size()));

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;

    wparams.prompt_tokens    = prompt_tokens.empty() ? nullptr : prompt_tokens.data();
    wparams.prompt_n_tokens  = prompt_tokens.empty() ? 0       : prompt_tokens.size();

    wparams.audio_ctx        = params.audio_ctx;
    wparams.speed_up         = params.speed_up;

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }

    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);

        result += text;

        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);

            prob += token.p;
            ++prob_n;
        }
    }

    if (prob_n > 0) {
        prob /= prob_n;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
	
	// print hex
	//printf("whisper result:\n");
	//for (char ch : result) printf("%X ", ch);
	//printf("\n\n");
				
    return result;
}

std::vector<std::string> get_words(const std::string &txt) {
    std::vector<std::string> words;

    std::istringstream iss(txt);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}


// @path full path to c:\\DATA\\LLM\\xtts\\xtts_play_allowed.txt
// @xtts_play_allowed: 0=dont play xtts, 1=xtts can play
void allow_xtts_file(std::string path, int xtts_play_allowed) {
    try{                                  
        const std::string fileName{path};
        std::ifstream readStream{fileName};
		std::string singleLine;
		bool doesExistAndIsReadable{readStream.good()};
		
		if(!doesExistAndIsReadable){
			//printf("%s file not found", path.c_str());
		}
		
		std::getline(readStream, singleLine);
		readStream.close();
		int stored_value = stoi(singleLine);
		if (stored_value != xtts_play_allowed)
		{
			std::ofstream writeStream{fileName};
			writeStream << xtts_play_allowed;
			writeStream.flush();
			writeStream.close();
			//printf("  written to file: %d\n", xtts_play_allowed);
		}
    } catch(...) {                          // exception handler
    }
}

// trim from start (in place)
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

bool IsPunctuationMark(char c) {
    switch (static_cast<unsigned char>(c)) {
        case ',':
            [[fallthrough]];
        case '.':
            [[fallthrough]];
        case '?':
            return true;
		case ':':
            return true;
		case '!':
            return true;	
        default:
            return false;
    }
}

std::string StripPunctuationMarks(const std::string& text) {
    std::string cleanText;
    for (const auto& c : text) {
        if (!IsPunctuationMark(c)) {
           cleanText += c;
        }
    }
    return cleanText;
}

std::string LowerCase(const std::string& text) {
    std::string lowerCasedText;
    for (const auto& c : text) {
        lowerCasedText += std::tolower(c, std::locale());
    }
    return lowerCasedText;
}

std::string ParseCommandAndGetKeyword(const std::string& textHeardTrimmed) {
    static const std::unordered_set<std::string> prefixNeedles{"google",  "google please", "please google", "can you google", "can you google", "Погугли", "По гугле", "угли", "углe", "По гугле пожалуйста", "По угли пожалуйста"};
    std::string sanitizedInput = LowerCase(StripPunctuationMarks(textHeardTrimmed));
    std::size_t pos = 0;
    bool startsWithPrefix = false;
    for(const auto& prefix : prefixNeedles) {
		if(sanitizedInput.compare(0,prefix.length(),prefix)==0) {
			pos = prefix.length();
			startsWithPrefix = true;
			break;
		}
	}
    if(!startsWithPrefix || pos==std::string::npos) {
        printf("unknown google command, trying anyway");
		pos = sanitizedInput.find("google");
		if (pos) pos = pos + 6;
		else pos = 0;
    }
    return sanitizedInput.substr(pos+1);
}

static size_t WriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
	((std::string*)userdata)->append((const char*)ptr, size * nmemb);
	return size * nmemb;
}

std::string RemoveTrailingCharacters(const std::string &inputString, const char targetCharacter) {
    auto lastNonTargetPosition = std::find_if(inputString.rbegin(), inputString.rend(), [targetCharacter](auto ch) {
        return ch != targetCharacter;
    });
    return std::string(inputString.begin(), lastNonTargetPosition.base());
}

std::string UrlEncode(const std::string& str) {
    CURL* curl = curl_easy_init();
    if (curl) {
        char* encodedUrl = curl_easy_escape(curl, str.c_str(), str.length());
        std::string escapedUrl(encodedUrl);
        curl_free(encodedUrl);
        curl_easy_cleanup(curl);
        return escapedUrl;
    }
    return {};
}

std::string send_curl_json(const std::string &url, const std::map<std::string, std::string>& params) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
	
    /* Initialize curl */
    curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }
    
    try {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());        
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
        
        /* Convert map to query string */
        std::ostringstream oss;
        bool firstParam = true;
		oss << "{";
        for (auto param : params) {
          if (!firstParam) oss<< ',';
          oss << "\"" << param.first << "\":\"" << param.second << "\"";
          firstParam=false;
        };
		oss << "}";
        fprintf(stdout, "send_curl_json: %s\n",oss.str().c_str());
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type:application/json"));
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, oss.str().c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);		
        res = curl_easy_perform(curl);
             
        if (res != CURLE_OK) {
            throw std::runtime_error(std::string("cURL error: ") + curl_easy_strerror(res));
        } else {
            //std::cout << "Request successful!" << std::endl;
        }
    }
	catch(...) {                          // exception handler
    }
	curl_easy_cleanup(curl);
	
	return readBuffer;
}


// simple curl
std::string send_curl(std::string url) {
    CURL *curl;
    CURLcode res; // Consider checking the result of curl_easy_perform with this
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl); // Use res to handle the result
        if (res != CURLE_OK) {
            fprintf(stderr, "CURL failed: %s\n", curl_easy_strerror(res));
        }
        curl_easy_cleanup(curl);
    }
    return readBuffer;
}

/* std::string socket_post(const std::string &url, const std::map<std::string, std::string>& params) { */
std::string socket_post(const std::string& hostname, const std::string& port, const std::string& path, const std::string& body) {
    int sock;
    struct addrinfo hints, *res, *p;
    int status;
    std::string response;

    // Setup hints
    std::memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // AF_INET or AF_INET6 to force version
    hints.ai_socktype = SOCK_STREAM;

    if ((status = getaddrinfo(hostname.c_str(), port.c_str(), &hints, &res)) != 0) {
        std::cerr << "getaddrinfo: " << gai_strerror(status) << std::endl;
        return "";
    }

    // Loop through all the results and connect to the first we can
    for(p = res; p != NULL; p = p->ai_next) {
        if ((sock = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
            perror("client: socket");
            continue;
        }

        if (connect(sock, p->ai_addr, p->ai_addrlen) == -1) {
            close(sock);
            perror("client: connect");
            continue;
        }

        break; // If we get here, we must have connected successfully
    }

    if (p == NULL) {
        // Loop through all the results and couldn't connect
        std::cerr << "client: failed to connect\n";
        return "";
    }

    freeaddrinfo(res); // All done with this structure

    std::string request = "POST " + path + " HTTP/1.1\r\n";
    request += "Host: " + hostname + "\r\n";
    request += "Content-Type: application/x-www-form-urlencoded\r\n";
    request += "Content-Length: " + std::to_string(body.length()) + "\r\n";
    request += "\r\n";
    request += body;

    // Send the request
    int bytes_sent = send(sock, request.c_str(), request.length(), 0);
    if (bytes_sent < 0) {
        perror("send");
        close(sock);
        return "";
    }

    // Receive data
    const int recv_buf_size = 4096; // Adjust as necessary
    char recv_buf[recv_buf_size];
    int bytes_received;
    while ((bytes_received = recv(sock, recv_buf, recv_buf_size - 1, 0)) > 0) {
        recv_buf[bytes_received] = '\0';
        response += std::string(recv_buf);
    }

    if (bytes_received < 0) {
        perror("recv");
    }

    close(sock);
    return response;
}


// async curl, but it's still blocking for some reason
// doesn't wait for responce
void send_tts_async(std::string text, std::string speaker_wav="emma_1", std::string language="en", std::string tts_url="http://localhost:8020/")
{
	text = ::replace(text, "...", ".");
	text = ::replace(text, "…", ".");
	text = ::replace(text, "??", "?");
	text = ::replace(text, "!!", "!");
	text = ::replace(text, "?!", "?");
	if (text.size() && text != "." && text != "," && text != "!" && text != "\n")
	{
		trim(text);
		text = ::replace(text, "\r", "");
		text = ::replace(text, "\n", " ");
		text = ::replace(text, "\"", "");
		tts_url= tts_url + "tts_to_audio/";
		//printf("send_tts_async sending, url: %s\n", tts_url.c_str());
		//for (char ch : tts_url) printf("%X ", ch);
		//printf("\n");
		
		
		CURL *http_handle;
		CURLM *multi_handle;
		int still_running = 1;
		curl_global_init(CURL_GLOBAL_DEFAULT);
		http_handle = curl_easy_init();
		std::string data = "{\"text\":\""+text+"\", \"language\":\""+language+"\", \"speaker_wav\":\""+speaker_wav+"\"}";
		//fprintf(stdout, " [data (%s)]\n", data.c_str());
		
		curl_easy_setopt(http_handle, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type:application/json"));
		curl_easy_setopt(http_handle, CURLOPT_URL, tts_url.c_str());
		curl_easy_setopt(http_handle, CURLOPT_POSTFIELDS, data.c_str());
		curl_easy_setopt(http_handle, CURLOPT_VERBOSE, 0L);
		
		std::string responseData;
		curl_easy_setopt(http_handle, CURLOPT_WRITEDATA, &responseData);
		curl_easy_setopt(http_handle, CURLOPT_WRITEFUNCTION, WriteCallback);

		multi_handle = curl_multi_init();
		curl_multi_add_handle(multi_handle, http_handle);

		do {
		  CURLMcode mc = curl_multi_perform(multi_handle, &still_running);
	 
		  if(still_running) mc = curl_multi_poll(multi_handle, NULL, 0, 1000, NULL);// wait for activity, timeout or "nothing" 
	 
		  if(mc)
			break;
		} while(still_running);

		//curl_multi_remove_handle(multi_handle, http_handle);
		curl_easy_cleanup(http_handle);
		curl_multi_cleanup(multi_handle);
		curl_global_cleanup();
		
	}
}

const std::string k_prompt_whisper = R"(A conversation with a person called {1}.)";

const std::string k_prompt_llama = R"(Text transcript of a never ending dialog, where {0} interacts with an AI assistant named {1}.
{1} is helpful, kind, honest, friendly, good at writing and never fails to answer {0}’s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {0} and {1} say aloud to each other.
The transcript only includes text, it does not include markup like HTML and Markdown.
{1} responds with short and concise answers.

{0}{4} Hello, {1}!
{1}{4} Hello {0}! How may I help you today?
{0}{4} What time is it?
{1}{4} It is {2} o'clock, {5}, year {3}.
{0}{4} What is a cat?
{1}{4} A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{0}{4} Name a color.
{1}{4} Blue
{0}{4})";

int run(int argc, const char ** argv) {
	whisper_params params;
	std::vector<std::thread> threads;
	std::thread t;
	int thread_i = 0;
	std::string text_to_speak_arr[100];
	bool last_output_has_username = false;
	
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }
	
	const std::string fileName{params.xtts_control_path};
	std::ifstream readStream{fileName};	
	if(!readStream.good()){
		printf("Warning: %s file not found, xtts wont stop on user speech without it\n", params.xtts_control_path.c_str());
	}
	readStream.close();

    // whisper init

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;

    struct whisper_context * ctx_wsp = whisper_init_from_file_with_params(params.model_wsp.c_str(), cparams);

    // llama init

    llama_backend_init(true);

    auto lmparams = llama_model_default_params();
    if (!params.use_gpu) {
        lmparams.n_gpu_layers = 0;
    } else {
        lmparams.n_gpu_layers = params.n_gpu_layers;
    }

    struct llama_model * model_llama = llama_load_model_from_file(params.model_llama.c_str(), lmparams);

    llama_context_params lcparams = llama_context_default_params();

    // tune these to your liking
    lcparams.n_ctx      = params.ctx_size; // 2048 default
    lcparams.seed       = 1;
    lcparams.n_threads  = params.n_threads;
    lcparams.n_batch  = 1024; // 512 is too small for init prompt

    struct llama_context * ctx_llama = llama_new_context_with_model(model_llama, lcparams);

    // print some info about the processing
    {
        fprintf(stderr, "\n");

        if (!whisper_is_multilingual(ctx_wsp)) {
			fprintf(stderr, "WARNING: model is not multilingual");
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }

    // init audio

    audio_async audio(30*1000);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    bool is_running  = true;
    bool force_speak = false;

    float prob0 = 0.0f;

    const std::string chat_symb = ":";

    std::vector<float> pcmf32_cur;
    std::vector<float> pcmf32_prompt;

	std::string prompt_whisper;
	if (params.language == "ru") std::string prompt_whisper = ::replace(k_prompt_whisper, "{1}", "Анна"); // Алиса is bad
	else std::string prompt_whisper = ::replace(k_prompt_whisper, "{1}", params.bot_name);

    // construct the initial prompt for LLaMA inference
    std::string prompt_llama = params.prompt.empty() ? k_prompt_llama : params.prompt;

    // need to have leading ' '
    prompt_llama.insert(0, 1, ' ');

    prompt_llama = ::replace(prompt_llama, "{0}", params.person);
    prompt_llama = ::replace(prompt_llama, "{1}", params.bot_name);

    {
        // get time string
        std::string time_str;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%H:%M", now);
            time_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{2}", time_str);
    }

    {
        // get year string
        std::string year_str;
        std::string ymd;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%Y", now);
            year_str = buf;
			strftime(buf, sizeof(buf), "%Y-%m-%d", now);
            ymd = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{3}", year_str);
        prompt_llama = ::replace(prompt_llama, "{5}", ymd);
    }

    prompt_llama = ::replace(prompt_llama, "{4}", chat_symb);

    // init session
    std::string path_session = params.path_session;
    std::vector<llama_token> session_tokens;
    auto embd_inp = ::llama_tokenize(ctx_llama, prompt_llama, true);

    if (!path_session.empty()) {
        fprintf(stderr, "%s: attempting to load saved session from %s\n", __func__, path_session.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(llama_n_ctx(ctx_llama));
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            for (size_t i = 0; i < session_tokens.size(); i++) {
                embd_inp[i] = session_tokens[i];
            }

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    // evaluate the initial prompt

    printf("\n");
    printf("%s : initializing - please wait ...\n", __func__);

	// Create and populate a llama_batch structure
	llama_batch batch;
	memset(&batch, 0, sizeof(batch)); // Initialize all fields to 0

	batch.n_tokens = embd_inp.size();
	batch.token = embd_inp.data(); // Assuming embd_inp is std::vector<llama_token>

	// Assuming you don't need other fields for this call
	// Now, call llama_decode with the batch
	if (llama_decode(ctx_llama, batch)) {
		fprintf(stderr, "%s : failed to eval\n", __func__);
		return 1;
	}

    if (params.verbose_prompt) {
        fprintf(stdout, "\n");
        fprintf(stdout, "%s", prompt_llama.c_str());
        fflush(stdout);
    }

     // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (n_matching_session_tokens >= embd_inp.size()) {
            fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
    }

    // HACK - because session saving incurs a non-negligible delay, for now skip re-saving session
    // if we loaded a session with at least 75% similarity. It's currently just used to speed up the
    // initial prompt so it doesn't need to be an exact match.
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < (embd_inp.size() * 3 / 4);

    printf("%s : done! start speaking in the microphone\n", __func__);

    // show wake command if enabled
    const std::string wake_cmd = params.wake_cmd;
    const int wake_cmd_length = get_words(wake_cmd).size();
    const bool use_wake_cmd = wake_cmd_length > 0;

    if (use_wake_cmd) {
        printf("%s : the wake-up command is: '%s%s%s'\n", __func__, "\033[1m", wake_cmd.c_str(), "\033[0m");
    }

    printf("\n");
    printf("%s%s", params.person.c_str(), chat_symb.c_str());
    fflush(stdout);

    // clear audio buffer
    audio.clear();

    // text inference variables
    /* const int voice_id = 2; */
    const int n_keep   = embd_inp.size();
    const int n_ctx    = llama_n_ctx(ctx_llama);

    int n_past = n_keep;
    int n_prev = 64;
    std::vector<int> past_prev_arr{};
    int n_past_prev = 0; // token count that was before the last answer
    int n_session_consumed = !path_session.empty() && session_tokens.size() > 0 ? session_tokens.size() : 0;
    std::vector<llama_token> embd;
	std::string text_heard_prev;
	std::string text_heard_trimmed;
	int new_command_allowed = 1;
	std::string google_resp;
	
	int last_command_time = 0;
	
    // reverse prompts for detecting when it's time to stop speaking
    std::vector<std::string> antiprompts = {
        params.person + chat_symb,
        "\n"
    };

    // main loop	
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // delay
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        int64_t t_ms = 0;

        {
            audio.get(2000, pcmf32_cur);
			// WHISPER_SAMPLE_RATE 16000
			//int vad_result = ::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1250, params.vad_thold, params.freq_thold, params.print_energy);
			int vad_result = ::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms, params.vad_thold, params.freq_thold, params.print_energy);
			if (vad_result == 1) // speech started
			{
				// user has started speaking, xtts cannot play
				//fprintf(stdout, "%s: Speech start! ...\n", __func__);
				allow_xtts_file(params.xtts_control_path, 0);
			}		
            if (vad_result >= 2 || force_speak)  // speech ended
			{
                //fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);
                audio.get(params.voice_ms, pcmf32_cur);
				
                std::string all_heard;

                if (!force_speak) {
                    all_heard = ::trim(::transcribe(ctx_wsp, params, pcmf32_cur, prompt_whisper, prob0, t_ms));
                }

                const auto words = get_words(all_heard);

                std::string wake_cmd_heard;
                std::string text_heard;

                for (int i = 0; i < (int) words.size(); ++i) {
                    if (i < wake_cmd_length) {
                        wake_cmd_heard += words[i] + " ";
                    } else {
                        text_heard += words[i] + " ";
                    }
                }				
				//fprintf(stdout, " [text_heard: (%s)]\n", text_heard.c_str());
				
				
                // check if audio starts with the wake-up command if enabled
                if (use_wake_cmd) {
                    const float sim = similarity(wake_cmd_heard, wake_cmd);

                    if ((sim < 0.7f) || (text_heard.empty())) {
                        audio.clear();
                        continue;
                    }
                }

                // optionally give audio feedback that the current text is being processed
				if (!params.heard_ok.empty()) {
					std::string speakCommand = "./speak_command"; // Ensure this script or executable is available and executable
					// Using params.heard_ok directly instead of an undefined 'argument'
					int ret = system((speakCommand + " '" + params.heard_ok + "'").c_str()); // Added single quotes to handle spaces in text
					if (ret != 0) {
						fprintf(stderr, "%s: failed to speak\n", __func__);
					}
				}
			
                // remove text between brackets using regex
                {
                    std::regex re("\\[.*?\\]");
                    text_heard = std::regex_replace(text_heard, re, "");
                }
				
                // remove text between brackets using regex
                {
                    std::regex re("\\(.*?\\)");
                    text_heard = std::regex_replace(text_heard, re, "");
                }
                // remove all characters, except for letters, numbers, punctuation and ':', '\'', '-', ' '
                if (params.language == "en") text_heard = std::regex_replace(text_heard, std::regex("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), ""); // breaks non latin text, e.g. Russian
                // take first line
                text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));

                // remove leading and trailing whitespace
                text_heard = std::regex_replace(text_heard, std::regex("^\\s+"), "");
                text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), "");

				// misheard text, sometimes whisper is hallucinating
				text_heard = RemoveTrailingCharacters(text_heard, '!');
				text_heard = RemoveTrailingCharacters(text_heard, ',');
				text_heard = RemoveTrailingCharacters(text_heard, '.');
				if (text_heard[0] == '.') text_heard.erase(0, 1);
				if (text_heard[0] == '!') text_heard.erase(0, 1);
				trim(text_heard);
				if (text_heard == "!" || text_heard == "." || text_heard == "Sil" || text_heard == "Bye" || text_heard == "Okay" || text_heard == "Okay." || text_heard == "Thank you." || text_heard == "Thank you" || text_heard == "Thanks." || text_heard == "Bye." || text_heard == "Thank you for listening." || text_heard == "К" || text_heard == "Спасибо" || text_heard == params.bot_name || text_heard == "*Звук!*" || text_heard == "Р" || text_heard.find("Редактор субтитров")!= std::string::npos || text_heard.find("можешь это сделать")!= std::string::npos || text_heard.find("Как дела?")!= std::string::npos || text_heard.find("Это")!= std::string::npos || text_heard.find("Добро пожаловать")!= std::string::npos) text_heard = "";
				text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), ""); // trailing whitespace

				
				//printf("Number of tokens in embd: %zu\n", embd.size());
				//printf("n_past_prev: %d\n", n_past_prev);
				//printf("text_heard_prev: %s\n", text_heard_prev);
				
				text_heard_trimmed = text_heard; // no periods or spaces
				trim(text_heard_trimmed);
				if (text_heard_trimmed[0] == '.') text_heard_trimmed.erase(0, 1);
				if (text_heard_trimmed[0] == '!') text_heard_trimmed.erase(0, 1);
				if (text_heard_trimmed[text_heard_trimmed.length() - 1] == '.' || text_heard_trimmed[text_heard_trimmed.length() - 1] == '!') text_heard_trimmed.erase(text_heard_trimmed.length() - 1, 1);
				trim(text_heard_trimmed);
				text_heard_trimmed = LowerCase(text_heard_trimmed); // not working right with utf and russian
				
				//fprintf(stdout, " [text_heard: (%s)]\n", text_heard.c_str());
				//fprintf(stdout, "text_heard_trimmed: %s%s%s", "\033[1m", text_heard_trimmed.c_str(), "\033[0m");
                fflush(stdout);
				std::string user_command;
				
				
				if (text_heard_trimmed.find("regenerate") != std::string::npos || text_heard_trimmed.find("Переделай") != std::string::npos || text_heard_trimmed.find("егенерируй") != std::string::npos || text_heard_trimmed.find("егенерировать") != std::string::npos) user_command = "regenerate";
				else if (text_heard_trimmed.find("google") != std::string::npos || text_heard_trimmed.find("Погугли") != std::string::npos || text_heard_trimmed.find("Пожалуйста, погугли") != std::string::npos || text_heard_trimmed.find("По гугл") != std::string::npos) user_command = "google";
				else if (text_heard_trimmed.find("reset") != std::string::npos || text_heard_trimmed.find("delete everything") != std::string::npos || text_heard_trimmed.find("Сброс") != std::string::npos || text_heard_trimmed.find("Сбросить") != std::string::npos || text_heard_trimmed.find("Удали все") != std::string::npos || text_heard_trimmed.find("Удалить все") != std::string::npos) user_command = "reset";
				else if (text_heard_trimmed.find("delete") != std::string::npos || text_heard_trimmed.find("please do it") != std::string::npos || text_heard_trimmed.find("Удали") != std::string::npos || text_heard_trimmed.find("Удалить сообщение") != std::string::npos || text_heard_trimmed.find("Удали сообщение") != std::string::npos || text_heard_trimmed.find("Удали два сообщения") != std::string::npos || text_heard_trimmed.find("Удали три сообщения") != std::string::npos) user_command = "delete";
				else if (text_heard_trimmed.find("stop") != std::string::npos || text_heard_trimmed.find("Стоп") != std::string::npos || text_heard_trimmed.find("Остановись") != std::string::npos || text_heard_trimmed.find("тановись") != std::string::npos || text_heard_trimmed.find("Хватит") != std::string::npos || text_heard_trimmed.find("Становись") != std::string::npos) user_command = "stop";
				
				// user has finished speaking, xtts can play
				allow_xtts_file(params.xtts_control_path, 1);
				
				if (user_command.size() && !new_command_allowed && std::time(0)-last_command_time >= 1) 
				{
					new_command_allowed = 1; // timeout before same command (whisper hallucinates sometimes)
					//printf("new_command_allowed: %d\n", new_command_allowed);
				}
				
				// REGEN
				if (user_command == "regenerate" || text_heard_trimmed == "Please regenerate" || text_heard_trimmed == "Regenerate please" || text_heard_trimmed == "Regenerate, please" || text_heard_trimmed == "Try again please" || text_heard_trimmed == "Try again, please" || text_heard_trimmed == "Please try again" || text_heard_trimmed == "Try again") 
				{
					if (new_command_allowed)
					{
						if (!past_prev_arr.empty())
						{
							// regenerate prev llama reply
							n_past_prev = past_prev_arr.back();
							past_prev_arr.pop_back();
							int rollback_num = embd_inp.size()-n_past_prev;
							if (rollback_num)
							{
								printf("regenerating %d tokens\n", rollback_num);						
								embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());						
								n_past -= rollback_num;
								text_heard = text_heard_prev;
								text_heard_trimmed = "";
								send_tts_async("Regenerating", params.xtts_voice, params.language, params.xtts_url);
								new_command_allowed = 0;
							}
						}						
					}
				}
				// DELETE
				else if (user_command == "delete" || text_heard_trimmed == "Please delete" || text_heard_trimmed == "Please delete the last message" || text_heard_trimmed == "Delete please" || text_heard_trimmed == "Delete, please") 
				{
					if (new_command_allowed) // delete prev user question and llama reply, then dont do anything
					{
						if (!past_prev_arr.empty())
						{
							if (text_heard_trimmed == "delete two messages" || text_heard_trimmed == "Удали 2 сообщения" || text_heard_trimmed == "Удали два сообщения")
							{
								n_past_prev = past_prev_arr.back();
								past_prev_arr.pop_back();
							}
							else if (text_heard_trimmed == "delete three messages" || text_heard_trimmed == "Удали 3 сообщения" || text_heard_trimmed == "Удали три сообщения")
							{
								n_past_prev = past_prev_arr.back();
								past_prev_arr.pop_back();
								n_past_prev = past_prev_arr.back();
								past_prev_arr.pop_back();
							}
						
							n_past_prev = past_prev_arr.back();
							past_prev_arr.pop_back();
							int rollback_num = embd_inp.size()-n_past_prev;
							if (rollback_num)
							{
								printf(" deleting %d tokens\n", rollback_num);						
								embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());						
								n_past -= rollback_num;
								text_heard = "";
								text_heard_trimmed = "";
								send_tts_async("Deleted", params.xtts_voice, params.language, params.xtts_url);
								last_command_time = std::time(0);
								new_command_allowed = 0;
							}
						}
						else 
						{
							printf("Nothing to delete more\n");
							send_tts_async("Nothing to delete more", "ux", params.language);
						}
					}
					audio.clear();
					//continue;
				}
				// RESET
				else if (user_command == "reset") 
				{
					if (new_command_allowed)
					{
						if (!past_prev_arr.empty())
						{
							// delete everything to start prompt
							n_past_prev = past_prev_arr.front();
							past_prev_arr.clear();
							int rollback_num = embd_inp.size()-n_past_prev;
							if (rollback_num)
							{
								printf(" Reset. deleting %d tokens\n", rollback_num);						
								embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());						
								n_past -= rollback_num;
								text_heard = "";
								text_heard_trimmed = "";
								send_tts_async("Reset whole context", params.xtts_voice, params.language, params.xtts_url);
								new_command_allowed = 0;
							}
						}
						else 
						{
							printf("Nothing to reset more\n");							
							send_tts_async("Nothing to reset more", params.xtts_voice, params.language, params.xtts_url);
						}
					}
					audio.clear();
					continue;
				}
				// STOP
				else if (user_command == "stop") 
				{
					printf(" Stopped!\n");
					audio.clear();
				}
				// GOOGLE
				else if (user_command == "google") 
				{
					std::string q = ParseCommandAndGetKeyword(text_heard_trimmed);
					if (q.size())
					{
						std::string url = params.google_url+"google?q="+UrlEncode(q);
						google_resp = send_curl(url);
						if (google_resp.size())
						{
							fprintf(stdout, "google_resp (%s): %s\n", q.c_str(), google_resp.c_str());
							if (google_resp.length() > 200) 
							{
								size_t space_pos = google_resp.find(' ', 201);   
								if(space_pos != std::string::npos) google_resp = google_resp.substr(0, space_pos);   
							}
							google_resp = "Google: "+google_resp+" .";
							text_heard += "\n"+google_resp;
							
							if (google_resp.size()) 
							{
								threads.emplace_back([&] // creates and starts a thread
								{
									if (google_resp.size()) send_tts_async(google_resp, params.xtts_voice, params.language, params.xtts_url);
								});
								thread_i++;
							}
						}
						printf("bad google resp for (%s), check that langchain server is ok",q.c_str());
					}
					else fprintf(stdout, "can't get search keyword from text_heard_trimmed: %s\n", text_heard_trimmed.c_str());
				}
				

                const std::vector<llama_token> tokens = llama_tokenize(ctx_llama, text_heard.c_str(), false);

                if (text_heard.empty() || tokens.empty() || force_speak) {
                    //fprintf(stdout, "%s: Heard nothing, skipping ...\n", __func__);
                    audio.clear();

                    continue;
                }
				printf("  [t: %zu] ", embd_inp.size());
                force_speak = false;

				text_heard_prev = text_heard;
				n_past_prev = embd_inp.size();
				past_prev_arr.push_back(embd_inp.size());
				
				//printf("text_heard_prev: %s\n", text_heard_prev);
                if (last_output_has_username) text_heard.insert(0, 1, ' ');
                else text_heard.insert(0, "\n"+params.person + chat_symb + " ");
                text_heard += "\n" + params.bot_name + chat_symb;
                fprintf(stdout, "%s%s%s", "\033[1m", text_heard.c_str(), "\033[0m");
                fflush(stdout);
				
                embd = ::llama_tokenize(ctx_llama, text_heard, false);

                // Append the new input tokens to the session_tokens vector
                if (!path_session.empty()) {
                    session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
                }
				
				if (threads.size() >=30)
				{
					printf("[!... %zu]\n", threads.size());
					for (auto& t : threads) 
					{
						try 
						{
							if (t.joinable()) t.join();							
							else printf("Notice: thread %d is NOT joinable\n", thread_i );
						}
						catch (const std::exception& ex) {
							std::cerr << "[Exception]: Failed join a thread: " << ex.what() << '\n';
						}
					}
					threads.clear();
					printf("]");
				}
				if (thread_i >= 80) thread_i = 0; // rotation
				
				

                // text inference
                bool done = false;
                std::string text_to_speak;
				int new_tokens = 0;	   
                while (true) {
                    // predict
					if (new_tokens > params.n_predict) break; // 64 default
					new_tokens++;
                    if (embd.size() > 0) {
                        if (n_past + (int) embd.size() > n_ctx) {
                            n_past = n_keep;

                            // insert n_left/2 tokens at the start of embd from last_n_tokens
                            embd.insert(embd.begin(), embd_inp.begin() + embd_inp.size() - n_prev, embd_inp.end());
                            // stop saving session if we run out of context
                            path_session = "";
                            //printf("\n---\n");
                            //printf("resetting: '");
                            //for (int i = 0; i < (int) embd.size(); i++) {
                            //    printf("%s", llama_token_to_piece(ctx_llama, embd[i]));
                            //}
                            //printf("'\n");
                            //printf("\n---\n");
                        }

                        // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
                        // REVIEW
                        if (n_session_consumed < (int) session_tokens.size()) {
                            size_t i = 0;
                            for ( ; i < embd.size(); i++) {
                                if (embd[i] != session_tokens[n_session_consumed]) {
                                    session_tokens.resize(n_session_consumed);
                                    break;
                                }

                                n_past++;
                                n_session_consumed++;

                                if (n_session_consumed >= (int) session_tokens.size()) {
                                    i++;
                                    break;
                                }
                            }
                            if (i > 0) {
                                embd.erase(embd.begin(), embd.begin() + i);
                            }
                        }

                        if (embd.size() > 0 && !path_session.empty()) {
                            session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                            n_session_consumed = session_tokens.size();
                        }

					// Assuming embd is a std::vector<llama_token> that needs to be passed to llama_decode
					llama_batch batch;
					memset(&batch, 0, sizeof(batch)); // Initialize all fields to 0

					batch.n_tokens = embd.size();
					batch.token = embd.data(); // Direct assignment from vector to pointer

					// Assuming you don't need other fields for this call
					// Now, call llama_decode with the batch
					if (llama_decode(ctx_llama, batch)) {
                            fprintf(stderr, "%s : failed to eval\n", __func__);
                            return 1;
                        }
                    }	
					

                    embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
                    n_past += embd.size();
					
                    embd.clear();
					new_command_allowed = 1;
                    if (done) break;

                    {
                        // out of user input, sample next token
                        const float top_k          = params.top_k;
                        const float top_p          = params.top_p;
                        const float temp           = params.temp;
                        const float repeat_penalty = params.repeat_penalty;

                        const int repeat_last_n    = 384; // was 256

                        if (!path_session.empty() && need_to_save_session) {
                            need_to_save_session = false;
                            llama_save_session_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.size());
                        }

                        llama_token id = 0;

                        {
                            auto logits = llama_get_logits(ctx_llama);
                            auto n_vocab = llama_n_vocab(model_llama);

                            logits[llama_token_eos(model_llama)] = 0;

                            std::vector<llama_token_data> candidates;
                            candidates.reserve(n_vocab);
                            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                            }

                            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                            // apply repeat penalty
                            const float nl_logit = logits[llama_token_nl(model_llama)];

                            llama_sample_repetition_penalties(ctx_llama, &candidates_p,
                                    embd_inp.data() + std::max(0, n_past - repeat_last_n),
                                    repeat_last_n, repeat_penalty, 0.0, 0.0f);

                            logits[llama_token_nl(model_llama)] = nl_logit;

                            if (temp <= 0) {
                                // Greedy sampling
                                id = llama_sample_token_greedy(ctx_llama, &candidates_p);
                            } else {
                                // Temperature sampling
                                llama_sample_top_k(ctx_llama, &candidates_p, top_k, 1);
                                llama_sample_top_p(ctx_llama, &candidates_p, top_p, 1);
                                llama_sample_temp (ctx_llama, &candidates_p, temp);
                                id = llama_sample_token(ctx_llama, &candidates_p);
                            }
                        }

                        if (id != llama_token_eos(model_llama)) {
                            // add it to the context
                            embd.push_back(id);

                            text_to_speak += llama_token_to_piece(ctx_llama, id);
                            //text_to_speak += llama_token_to_piece(ctx_llama, id).c_str();

                            printf("%s", llama_token_to_piece(ctx_llama, id).c_str());
							
							// streaming tts
							int text_len = text_to_speak.size();
							if (text_len >= 10 && (text_to_speak[text_len-1] == '.' /*|| text_to_speak[text_len-1] == ','*/ || text_to_speak[text_len-1] == '?' || text_to_speak[text_len-1] == '!' || text_to_speak[text_len-1] == ';' || text_to_speak[text_len-1] == ':'))
							{
								text_to_speak = ::replace(text_to_speak, "\"", "'");
								text_to_speak = ::replace(text_to_speak, antiprompts[0], "");
								
								if (text_to_speak.size()) // first and mid parts of the sentence
								{
									// system TTS
									//int ret = system(("start /B "+params.speak + " " + std::to_string(voice_id) + " \"" + text_to_speak + "\" & exit").c_str()); // for windows
									//int ret = system((params.speak + " " + std::to_string(voice_id) + " \"" + text_to_speak + "\" &").c_str()); // for linux
								
									// XTTS in threads
									text_to_speak_arr[thread_i] = text_to_speak;
									try 
									{
										threads.emplace_back([&] // creates and starts a thread. crashes after 100 threads
										{
											if (text_to_speak_arr[thread_i-1].size())
											{
												send_tts_async(text_to_speak_arr[thread_i-1], params.xtts_voice, params.language, params.xtts_url);
												text_to_speak_arr[thread_i-1] = "";
											}
										});
										thread_i++;
										text_to_speak = "";  
									}
									catch (const std::exception& ex) {
										std::cerr << "[Exception]: Failed to push_back mid thread: " << ex.what() << '\n';
									}
									
									// check if user is speaking after each sentence. Good for 'stop' command, but it doesn't call whisper recognition. 'Stop' -> wait 1s -> ask question
									audio.get(500, pcmf32_cur); // 0.5s?
									int vad_result = ::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms, params.vad_thold, params.freq_thold, params.print_energy);
									if (vad_result == 1) // speech started
									{
										// user has started speaking, xtts cannot play
										fprintf(stdout, " [Speech detected! Aborting ...]\n");
										allow_xtts_file(params.xtts_control_path, 0);
										done = true; // generation stops
										break;
									}
								}
							}			   
                        }
                    }
					
                    {
                        std::string last_output;
                        for (int i = embd_inp.size() - 16; i < (int) embd_inp.size(); i++) {
                            last_output += llama_token_to_piece(ctx_llama, embd_inp[i]);
                        }
                        last_output += llama_token_to_piece(ctx_llama, embd[0]);

						int i_antiprompt = 0;
						last_output_has_username = false;
                        for (std::string & antiprompt : antiprompts) 
						{
                            if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                                done = true;
                                text_to_speak = ::replace(text_to_speak, antiprompt, "");
                                fflush(stdout);
                                need_to_save_session = true;
								if (i_antiprompt == 0) 
								{
									last_output_has_username = true;
								}
								//printf("antiprompt: %s\n", antiprompt.c_str());
                                break;
                            }
							i_antiprompt++;
                        }
                    }

                    is_running = sdl_poll_events();

                    if (!is_running) {
                        break;
                    }
                }
				
				// final part of the sentence, if any
                text_to_speak = ::replace(text_to_speak, "\"", "'");				
				if (text_to_speak.size()) 
				{
					text_to_speak_arr[thread_i] = text_to_speak;
					try 
					{		
						threads.emplace_back([&] // creates and starts a thread
						{
							if (text_to_speak_arr[thread_i-1].size())
							{
								send_tts_async(text_to_speak_arr[thread_i-1], params.xtts_voice, params.language, params.xtts_url);
								text_to_speak_arr[thread_i-1] = "";
							}
						});	
						thread_i++;
						text_to_speak = "";  						
					}
					catch (const std::exception& ex) {
						std::cerr << "[Exception]: Failed to emplace fin thread: " << ex.what() << '\n'; 
					}
				}
				
                audio.clear();
            }
        }
    }

    audio.pause();

    whisper_print_timings(ctx_wsp);
    whisper_free(ctx_wsp);

    llama_print_timings(ctx_llama);
    llama_free(ctx_llama);

    return 0;
}

//#ifdef WIN32
//	// Initialize Winsock
//	WORD ver = MAKEWORD(2 , 2);
//	WSADATA dat;
//    int wsastartup = WSAStartup(ver, &dat);
//#endif
	
#if _WIN32
int wmain(int argc, const wchar_t ** argv_UTF16LE) {
	//setlocale(LC_ALL,"Russian");
	console::init(true, true);
    atexit([]() { console::cleanup(); });
    std::vector<std::string> buffer(argc);
    std::vector<const char*> argv_UTF8(argc);
	for (int i = 0; i < argc; ++i) {
        buffer[i] = console::UTF16toUTF8(argv_UTF16LE[i]);
        argv_UTF8[i] = buffer[i].c_str();
    }
    return run(argc, argv_UTF8.data());
}
#else
int main(int argc, char** argv) {
    // Linux initialization code here (if needed)

    // Convert argv to std::vector<std::string> if necessary
    std::vector<std::string> args(argv, argv + argc);
    
    // Example conversion if needed (adjust based on actual usage in your program)
    std::vector<const char*> argv_converted(argc);
    for (size_t i = 0; i < args.size(); ++i) {
        argv_converted[i] = args[i].c_str();
    }

    // Call your main logic here, adjust function signature as needed
    return run(argc, argv_converted.data());
}
#endif

