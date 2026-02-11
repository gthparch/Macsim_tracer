//
// This code compresses bin_trace_i.raw files, which are uncompressed binary files, into compressed trace_i.raw files
//
#include <assert.h>
#include <iostream>
#include <fstream>
#include <zlib.h>
#include <dirent.h>
#include <vector>
#include <cstring>

#define CHUNK_SIZE 16384

// std::string trace_path = "/fast_data/echung67/trace/nvbit/temp/";
std::string trace_path = "./"; // FIXME!!

std::vector<std::string> listDirectories(const std::string& path)
{
    std::vector<std::string> directories;
    DIR* dir = opendir(path.c_str());
    if (dir != nullptr)
    {
        dirent* entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            // Check if the entry is a directory (DT_DIR)
            if (entry->d_type == DT_DIR)
            {
                // Ignore "." and ".." directories
                if (std::string(entry->d_name) != "." && std::string(entry->d_name) != "..")
                {
                    directories.push_back(std::string(entry->d_name));
                }
            }
        }
        closedir(dir);
    }
    return directories;
}

int main() {
    DIR *dir;
    struct dirent *ent;
    std::vector<std::string> dirnames = listDirectories(trace_path);
    std::vector<std::vector<std::string>> filenames;
    for (size_t i=0; i<dirnames.size(); i++){
        std::vector<std::string> filename;
        filenames.push_back(filename);
    }
    int i=0;
    std::cout << dirnames.size() << std::endl;
    for (const std::string& ker : dirnames) { // the name of a directory is the same as the name of a kernel
        if ((dir = opendir((trace_path + ker + "/").c_str())) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                if (std::string(ent->d_name).find("bin_trace_") == 0 &&
                    std::string(ent->d_name).find(".raw") == std::string(ent->d_name).size() - 4) {
                    filenames[i].push_back(ent->d_name);
                }
            }
            closedir(dir);
        } else {
            std::cerr << "Error opening directory\n";
            return 1;
        }
        for (auto file : filenames[i]) {
            std::ifstream input_file(trace_path + ker + "/" + file, std::ios::binary);
            std::string orig_file_path = trace_path + ker + "/" + file;
            if (!input_file) {
                std::cerr << "Error opening input file: " << trace_path + ker + "/" + file << "\n";
                assert(0);
            }
            file.erase(0, 4); // remove first 4 characters ("bin_")
            std::string output_filepath = trace_path + ker + "/" + file;
            gzFile output_file = gzopen(output_filepath.c_str(), "wb");
            if (output_file == NULL) {
                std::cerr << "Error opening output file: " << output_filepath << "\n";
                assert(0);
            }

            unsigned char buffer[CHUNK_SIZE];
            int bytes_read;
            while ((bytes_read = input_file.read(reinterpret_cast<char*>(buffer), CHUNK_SIZE).gcount()) > 0) {
                int bytes_written = gzwrite(output_file, buffer, bytes_read);
                if (bytes_written == 0) {
                    std::cerr << "Error writing to output file: " << output_filepath << "\n";
                    assert(0);
                }
            }

            gzclose(output_file);
            input_file.close();

            // Delete the bin_* file
            
            if (std::remove(orig_file_path.c_str()) != 0) {
                std::perror("Error deleting file");
                assert(0);
            } 
            // else {
            //     std::cout << "File deleted successfully\n";
            // }
        }
        i++;
    }
    return 0;
}
