#include "experiments/harrington1d_code_capacity.h"
#include "experiments/harrington1d_phenomenological.h"
#include "experiments/scala1d_code_capacity.h"
#include "experiments/scala1d_phenomenological.h"

#include "experiments/lifetime_utils.h"
#include "experiments/outcome_utils.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string model = "scala1d";
    std::string noise_model = "phenomenological";

    std::size_t distance = 7;
    int samples = 100;
    int max_steps = 100000;
    int reset_time = 16;

    double p_data = 0.02;
    double p_meas = 0.0;
    double p_signal = 0.0;

    // Harrington-specific parameters
    int U = 16;
    double fN = 0.4;
    double fC = 0.9;

    unsigned int seed = 12345;
    std::string output;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --model <scala1d|harrington1d>\n"
        << "  --noise-model <phenomenological|code_capacity>\n"
        << "  --distance <int>\n"
        << "  --samples <int>\n"
        << "  --max-steps <int>\n"
        << "  --reset-time <int>\n"
        << "  --p-data <float>\n"
        << "  --p-meas <float>\n"
        << "  --p-signal <float>\n"
        << "  --U <int>\n"
        << "  --fN <float>\n"
        << "  --fC <float>\n"
        << "  --seed <int>\n"
        << "  --output <path.npy>\n"
        << "  --help\n";
}

Args parse_args(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        auto require_value = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for option " + option);
            }
            return argv[++i];
        };

        if (key == "--model") {
            args.model = require_value(key);
        } else if (key == "--noise-model") {
            args.noise_model = require_value(key);
        } else if (key == "--distance") {
            args.distance = static_cast<std::size_t>(std::stoul(require_value(key)));
        } else if (key == "--samples") {
            args.samples = std::stoi(require_value(key));
        } else if (key == "--max-steps") {
            args.max_steps = std::stoi(require_value(key));
        } else if (key == "--reset-time") {
            args.reset_time = std::stoi(require_value(key));
        } else if (key == "--p-data") {
            args.p_data = std::stod(require_value(key));
        } else if (key == "--p-meas") {
            args.p_meas = std::stod(require_value(key));
        } else if (key == "--p-signal") {
            args.p_signal = std::stod(require_value(key));
        } else if (key == "--U") {
            args.U = std::stoi(require_value(key));
        } else if (key == "--fN") {
            args.fN = std::stod(require_value(key));
        } else if (key == "--fC") {
            args.fC = std::stod(require_value(key));
        } else if (key == "--seed") {
            args.seed = static_cast<unsigned int>(std::stoul(require_value(key)));
        } else if (key == "--output") {
            args.output = require_value(key);
        } else if (key == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }

    return args;
}

void write_metadata_scala1d_phenomenological(
    const Args& args,
    const std::string& meta_path,
    double mean
) {
    std::ofstream meta(meta_path);
    meta << "{\n";
    meta << "  \"model\": \"scala1d\",\n";
    meta << "  \"noise_model\": \"phenomenological\",\n";
    meta << "  \"observable\": \"lifetime\",\n";
    meta << "  \"distance\": " << args.distance << ",\n";
    meta << "  \"samples\": " << args.samples << ",\n";
    meta << "  \"max_steps\": " << args.max_steps << ",\n";
    meta << "  \"reset_time\": " << args.reset_time << ",\n";
    meta << "  \"p_data\": " << args.p_data << ",\n";
    meta << "  \"p_meas\": " << args.p_meas << ",\n";
    meta << "  \"p_signal\": " << args.p_signal << ",\n";
    meta << "  \"seed\": " << args.seed << ",\n";
    meta << "  \"mean_lifetime\": " << mean << ",\n";
    meta << "  \"data_file\": \"" << args.output << "\"\n";
    meta << "}\n";
}

void write_metadata_scala1d_code_capacity(
    const Args& args,
    const std::string& meta_path,
    double p_logical
) {
    std::ofstream meta(meta_path);
    meta << "{\n";
    meta << "  \"model\": \"scala1d\",\n";
    meta << "  \"noise_model\": \"code_capacity\",\n";
    meta << "  \"observable\": \"logical_error_rate\",\n";
    meta << "  \"distance\": " << args.distance << ",\n";
    meta << "  \"samples\": " << args.samples << ",\n";
    meta << "  \"max_steps\": " << args.max_steps << ",\n";
    meta << "  \"p_data\": " << args.p_data << ",\n";
    meta << "  \"seed\": " << args.seed << ",\n";
    meta << "  \"logical_error_rate\": " << p_logical << ",\n";
    meta << "  \"data_file\": \"" << args.output << "\"\n";
    meta << "}\n";
}

void write_metadata_harrington1d_phenomenological(
    const Args& args,
    const std::string& meta_path,
    double mean
) {
    std::ofstream meta(meta_path);
    meta << "{\n";
    meta << "  \"model\": \"harrington1d\",\n";
    meta << "  \"noise_model\": \"phenomenological\",\n";
    meta << "  \"observable\": \"lifetime\",\n";
    meta << "  \"distance\": " << args.distance << ",\n";
    meta << "  \"samples\": " << args.samples << ",\n";
    meta << "  \"max_steps\": " << args.max_steps << ",\n";
    meta << "  \"p_data\": " << args.p_data << ",\n";
    meta << "  \"p_meas\": " << args.p_meas << ",\n";
    meta << "  \"U\": " << args.U << ",\n";
    meta << "  \"fN\": " << args.fN << ",\n";
    meta << "  \"fC\": " << args.fC << ",\n";
    meta << "  \"seed\": " << args.seed << ",\n";
    meta << "  \"mean_lifetime\": " << mean << ",\n";
    meta << "  \"data_file\": \"" << args.output << "\"\n";
    meta << "}\n";
}

void write_metadata_harrington1d_code_capacity(
    const Args& args,
    const std::string& meta_path,
    double p_logical
) {
    std::ofstream meta(meta_path);
    meta << "{\n";
    meta << "  \"model\": \"harrington1d\",\n";
    meta << "  \"noise_model\": \"code_capacity\",\n";
    meta << "  \"observable\": \"logical_error_rate\",\n";
    meta << "  \"distance\": " << args.distance << ",\n";
    meta << "  \"samples\": " << args.samples << ",\n";
    meta << "  \"max_steps\": " << args.max_steps << ",\n";
    meta << "  \"p_data\": " << args.p_data << ",\n";
    meta << "  \"U\": " << args.U << ",\n";
    meta << "  \"fN\": " << args.fN << ",\n";
    meta << "  \"fC\": " << args.fC << ",\n";
    meta << "  \"seed\": " << args.seed << ",\n";
    meta << "  \"logical_error_rate\": " << p_logical << ",\n";
    meta << "  \"data_file\": \"" << args.output << "\"\n";
    meta << "}\n";
}

bool is_power_of_three(std::size_t n)
{
    if (n == 0) {
        return false;
    }

    while (n % 3 == 0) {
        n /= 3;
    }

    return n == 1;
}

}  // namespace

int main(int argc, char* argv[])
{
    try {
        const Args args = parse_args(argc, argv);

		if (args.reset_time > static_cast<int>(args.distance)) {
			throw std::runtime_error("--reset-time must be less than or equal to --distance");
		}

        if (args.model == "scala1d") {
            if (args.noise_model == "phenomenological") {
                const auto lifetimes = run_scala1d_phenomenological_lifetimes(
                    args.distance,
                    args.samples,
                    args.max_steps,
                    args.reset_time,
                    args.p_data,
                    args.p_meas,
                    args.p_signal,
                    args.seed
                );

                const double mean = mean_lifetime(lifetimes);

                if (!args.output.empty()) {
                    write_lifetimes_npy(lifetimes, args.output);
                    std::filesystem::path p(args.output);
                    const std::string meta_path = p.replace_extension(".json").string();
                    write_metadata_scala1d_phenomenological(args, meta_path, mean);
                }

                std::cout << "model scala1d\n";
                std::cout << "noise_model phenomenological\n";
                std::cout << "mean_lifetime " << mean << '\n';
                std::cout << "num_samples " << lifetimes.size() << '\n';
                if (!args.output.empty()) {
                    std::cout << "output " << args.output << '\n';
                }
                return 0;
            }

            if (args.noise_model == "code_capacity") {
                if (args.p_meas != 0.0) {
                    throw std::runtime_error("scala1d code_capacity does not support measurement noise");
                }
                if (args.p_signal != 0.0) {
                    throw std::runtime_error("scala1d code_capacity does not support signal noise");
                }

                const auto outcomes = run_scala1d_code_capacity(
                    args.distance,
                    args.samples,
                    args.max_steps,
                    args.p_data,
                    args.seed
                );

                const double p_logical = logical_error_rate(outcomes);

                if (!args.output.empty()) {
                    write_outcomes_npy(outcomes, args.output);
                    std::filesystem::path p(args.output);
                    const std::string meta_path = p.replace_extension(".json").string();
                    write_metadata_scala1d_code_capacity(args, meta_path, p_logical);
                }

                std::cout << "model scala1d\n";
                std::cout << "noise_model code_capacity\n";
                std::cout << "logical_error_rate " << p_logical << '\n';
                std::cout << "num_samples " << outcomes.size() << '\n';
                if (!args.output.empty()) {
                    std::cout << "output " << args.output << '\n';
                }
                return 0;
            }

            throw std::runtime_error("Unsupported noise model for scala1d: " + args.noise_model);
        }

		if (args.model == "harrington1d") {

			if (!is_power_of_three(args.distance)) {
				throw std::runtime_error("harrington1d requires --distance to be a power of 3");
			}

			if (args.noise_model == "phenomenological") {
				if (args.p_signal != 0.0) {
					throw std::runtime_error("harrington1d does not use p_signal");
				}

				const auto lifetimes = run_harrington1d_phenomenological_lifetimes(
					args.distance,
					args.samples,
					args.max_steps,
					args.p_data,
					args.p_meas,
					args.U,
					args.fN,
					args.fC,
					args.seed
				);

				const double mean = mean_lifetime(lifetimes);

				if (!args.output.empty()) {
					write_lifetimes_npy(lifetimes, args.output);
					std::filesystem::path p(args.output);
					const std::string meta_path = p.replace_extension(".json").string();
					write_metadata_harrington1d_phenomenological(args, meta_path, mean);
				}

				std::cout << "model harrington1d\n";
				std::cout << "noise_model phenomenological\n";
				std::cout << "mean_lifetime " << mean << '\n';
				std::cout << "num_samples " << lifetimes.size() << '\n';
				if (!args.output.empty()) {
					std::cout << "output " << args.output << '\n';
				}
				return 0;
			}

			if (args.noise_model == "code_capacity") {
				if (args.p_meas != 0.0) {
					throw std::runtime_error("harrington1d code_capacity does not support measurement noise");
				}
				if (args.p_signal != 0.0) {
					throw std::runtime_error("harrington1d code_capacity does not use p_signal");
				}

				const auto outcomes = run_harrington1d_code_capacity(
					args.distance,
					args.samples,
					args.max_steps,
					args.p_data,
					args.U,
					args.fN,
					args.fC,
					args.seed
				);

				const double p_logical = logical_error_rate(outcomes);

				if (!args.output.empty()) {
					write_outcomes_npy(outcomes, args.output);
					std::filesystem::path p(args.output);
					const std::string meta_path = p.replace_extension(".json").string();
					write_metadata_harrington1d_code_capacity(args, meta_path, p_logical);
				}

				std::cout << "model harrington1d\n";
				std::cout << "noise_model code_capacity\n";
				std::cout << "logical_error_rate " << p_logical << '\n';
				std::cout << "num_samples " << outcomes.size() << '\n';
				if (!args.output.empty()) {
					std::cout << "output " << args.output << '\n';
				}
				return 0;
			}

			throw std::runtime_error("Unsupported noise model for harrington1d: " + args.noise_model);
		}

        throw std::runtime_error(
            "Unsupported model: " + args.model +
            ". Expected 'scala1d' or 'harrington1d'."
        );

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
