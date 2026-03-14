//
// Parallel compression of bin_trace_i.raw → trace_i.raw (gzip)
// Uses std::thread pool for file-level parallelism.
//
#include <assert.h>
#include <iostream>
#include <fstream>
#include <zlib.h>
#include <dirent.h>
#include <vector>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>

#define CHUNK_SIZE 16384
#define DEFAULT_THREADS 32

std::string trace_path = "./";
std::atomic<int> files_done{0};
int files_total = 0;

struct CompressTask {
    std::string input_path;   // bin_trace_*.raw
    std::string output_path;  // trace_*.raw
};

void compress_worker(std::vector<CompressTask>& tasks,
                     std::atomic<int>& next_task) {
    unsigned char buffer[CHUNK_SIZE];
    while (true) {
        int idx = next_task.fetch_add(1);
        if (idx >= (int)tasks.size()) break;

        const auto& t = tasks[idx];
        std::ifstream input_file(t.input_path, std::ios::binary);
        if (!input_file) {
            std::cerr << "Error opening: " << t.input_path << "\n";
            continue;
        }

        gzFile output_file = gzopen(t.output_path.c_str(), "wb");
        if (output_file == NULL) {
            std::cerr << "Error opening: " << t.output_path << "\n";
            continue;
        }

        int bytes_read;
        while ((bytes_read = input_file.read(
                    reinterpret_cast<char*>(buffer), CHUNK_SIZE).gcount()) > 0) {
            int bytes_written = gzwrite(output_file, buffer, bytes_read);
            if (bytes_written == 0) {
                std::cerr << "Error writing: " << t.output_path << "\n";
                break;
            }
        }

        gzclose(output_file);
        input_file.close();
        std::remove(t.input_path.c_str());

        int done = files_done.fetch_add(1) + 1;
        if (done % 200 == 0 || done == files_total)
            fprintf(stderr, "\r[compress] %d / %d files", done, files_total);
    }
}

std::vector<std::string> listDirectories(const std::string& path) {
    std::vector<std::string> directories;
    DIR* dir = opendir(path.c_str());
    if (dir != nullptr) {
        dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_DIR &&
                std::string(entry->d_name) != "." &&
                std::string(entry->d_name) != "..") {
                directories.push_back(std::string(entry->d_name));
            }
        }
        closedir(dir);
    }
    return directories;
}

int main() {
    std::vector<std::string> dirnames = listDirectories(trace_path);
    std::cout << dirnames.size() << std::endl;

    /* Collect all compression tasks */
    std::vector<CompressTask> tasks;
    for (const std::string& ker : dirnames) {
        DIR* dir = opendir((trace_path + ker + "/").c_str());
        if (!dir) continue;
        dirent* ent;
        while ((ent = readdir(dir)) != NULL) {
            std::string fname(ent->d_name);
            if (fname.find("bin_trace_") == 0 &&
                fname.size() > 4 &&
                fname.substr(fname.size() - 4) == ".raw") {
                CompressTask t;
                t.input_path = trace_path + ker + "/" + fname;
                std::string out_fname = fname.substr(4); // remove "bin_"
                t.output_path = trace_path + ker + "/" + out_fname;
                tasks.push_back(t);
            }
        }
        closedir(dir);
    }

    files_total = (int)tasks.size();

    /* Use hardware concurrency, fall back to DEFAULT_THREADS */
    int hw_threads = (int)std::thread::hardware_concurrency();
    int nthreads = std::min(hw_threads > 0 ? hw_threads : DEFAULT_THREADS, files_total);
    fprintf(stderr, "[compress] %d files, %d threads\n", files_total, nthreads);

    /* Launch thread pool */
    std::atomic<int> next_task{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++) {
        threads.emplace_back(compress_worker, std::ref(tasks), std::ref(next_task));
    }
    for (auto& th : threads) {
        th.join();
    }

    fprintf(stderr, "\n[compress] Done.\n");
    return 0;
}
