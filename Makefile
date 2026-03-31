CXX := c++
PYTHON := python3

CXXFLAGS := -O3 -std=c++20 -Icpp/include
BUILD_DIR := build
PY_NATIVE_DIR := python/_native

PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)

MAIN_EXE := $(BUILD_DIR)/ca_run
HARRINGTON2D_MODULE := $(PY_NATIVE_DIR)/harrington2d$(PYTHON_EXT_SUFFIX)

SCALA2D_MODULE := $(PY_NATIVE_DIR)/scala2d$(PYTHON_EXT_SUFFIX)

COMMON_SRCS := \
	cpp/src/codes/repetition_code.cpp \
	cpp/src/noise/measurement_noise.cpp \
	cpp/src/models/scala1d.cpp \
	cpp/src/models/harrington1d_memory.cpp \
	cpp/src/models/harrington1d_cell.cpp \
	cpp/src/models/harrington1d.cpp \
	cpp/src/experiments/lifetime_utils.cpp \
	cpp/src/experiments/outcome_utils.cpp \
	cpp/src/experiments/scala1d_phenomenological.cpp \
	cpp/src/experiments/scala1d_code_capacity.cpp \
	cpp/src/experiments/harrington1d_phenomenological.cpp \
	cpp/src/experiments/harrington1d_code_capacity.cpp

MAIN_SRC := cpp/src/main.cpp

TEST_REP_SRC := cpp/tests/test_repetition_code.cpp
TEST_HARR_CONSTR_SRC := cpp/tests/test_harrington1d_constructor.cpp
TEST_SCALA_CC_SRC := cpp/tests/test_scala1d_code_capacity.cpp
TEST_HARR_CC_SRC := cpp/tests/test_harrington1d_code_capacity.cpp

TEST_REP_EXE := $(BUILD_DIR)/test_repetition_code
TEST_HARR_CONSTR_EXE := $(BUILD_DIR)/test_harrington1d_constructor
TEST_SCALA_CC_EXE := $(BUILD_DIR)/test_scala1d_code_capacity
TEST_HARR_CC_EXE := $(BUILD_DIR)/test_harrington1d_code_capacity

.PHONY: all dirs clean tests harrington2d \
	scan-scala1d-pheno-small scan-harrington1d-pheno-small \
	summarize-pheno-small plot-tf-vs-p \
	summarize-scala1d-pheno summarize-scala1d-cc \
	summarize-harrington1d-pheno summarize-harrington1d-cc \
	scan-harrington2d-cc scan-harrington2d-pheno \
	summarize-harrington2d-cc summarize-harrington2d-pheno \
	plot-harrington2d \
	scala2d \
	scan-scala2d-cc scan-scala2d-pheno \
	summarize-scala2d-cc summarize-scala2d-pheno \
	plot-scala2d test-scala2d \
	plot-scala1d-pheno-paper plot-scala2d-pheno-paper plot-scala2d-cc-paper

all: $(MAIN_EXE)

dirs:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(PY_NATIVE_DIR)

$(MAIN_EXE): dirs $(COMMON_SRCS) $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) $(COMMON_SRCS) $(MAIN_SRC) -o $(MAIN_EXE)

$(TEST_REP_EXE): dirs cpp/src/codes/repetition_code.cpp $(TEST_REP_SRC)
	$(CXX) $(CXXFLAGS) cpp/src/codes/repetition_code.cpp $(TEST_REP_SRC) -o $(TEST_REP_EXE)

$(TEST_HARR_CONSTR_EXE): dirs \
	cpp/src/models/harrington1d_memory.cpp \
	cpp/src/models/harrington1d_cell.cpp \
	cpp/src/models/harrington1d.cpp \
	$(TEST_HARR_CONSTR_SRC)
	$(CXX) $(CXXFLAGS) \
		cpp/src/models/harrington1d_memory.cpp \
		cpp/src/models/harrington1d_cell.cpp \
		cpp/src/models/harrington1d.cpp \
		$(TEST_HARR_CONSTR_SRC) \
		-o $(TEST_HARR_CONSTR_EXE)

$(TEST_SCALA_CC_EXE): dirs \
	cpp/src/codes/repetition_code.cpp \
	cpp/src/models/scala1d.cpp \
	cpp/src/experiments/outcome_utils.cpp \
	cpp/src/experiments/scala1d_code_capacity.cpp \
	$(TEST_SCALA_CC_SRC)
	$(CXX) $(CXXFLAGS) \
		cpp/src/codes/repetition_code.cpp \
		cpp/src/models/scala1d.cpp \
		cpp/src/experiments/outcome_utils.cpp \
		cpp/src/experiments/scala1d_code_capacity.cpp \
		$(TEST_SCALA_CC_SRC) \
		-o $(TEST_SCALA_CC_EXE)

$(TEST_HARR_CC_EXE): dirs \
	cpp/src/codes/repetition_code.cpp \
	cpp/src/models/harrington1d_memory.cpp \
	cpp/src/models/harrington1d_cell.cpp \
	cpp/src/models/harrington1d.cpp \
	cpp/src/experiments/outcome_utils.cpp \
	cpp/src/experiments/harrington1d_code_capacity.cpp \
	$(TEST_HARR_CC_SRC)
	$(CXX) $(CXXFLAGS) \
		cpp/src/codes/repetition_code.cpp \
		cpp/src/models/harrington1d_memory.cpp \
		cpp/src/models/harrington1d_cell.cpp \
		cpp/src/models/harrington1d.cpp \
		cpp/src/experiments/outcome_utils.cpp \
		cpp/src/experiments/harrington1d_code_capacity.cpp \
		$(TEST_HARR_CC_SRC) \
		-o $(TEST_HARR_CC_EXE)

tests: $(TEST_REP_EXE) $(TEST_HARR_CONSTR_EXE) $(TEST_SCALA_CC_EXE) $(TEST_HARR_CC_EXE)
	./$(TEST_REP_EXE)
	./$(TEST_HARR_CONSTR_EXE)
	./$(TEST_SCALA_CC_EXE)
	./$(TEST_HARR_CC_EXE)

harrington2d: dirs
	$(CXX) -O3 -std=c++20 -shared -fPIC \
	  -undefined dynamic_lookup \
	  $(PYBIND11_INCLUDES) \
	  cpp/src/models/harrington2d_memory.cpp \
	  cpp/src/models/harrington2d_cell.cpp \
	  cpp/src/models/harrington2d.cpp \
	  cpp/src/bindings/harrington2d_module_bindings.cpp \
	  -Icpp/include \
	  -o $(HARRINGTON2D_MODULE)

scala2d: dirs
	$(CXX) -O3 -std=c++20 -shared -fPIC \
	  -undefined dynamic_lookup \
	  $(PYBIND11_INCLUDES) \
	  cpp/src/models/scala2d_cell.cpp \
	  cpp/src/models/scala2d.cpp \
	  cpp/src/bindings/scala2d_module_bindings.cpp \
	  -Icpp/include \
	  -o $(SCALA2D_MODULE)

scan-scala1d-pheno-small:
	python scripts/scan.py --config configs/scala1d_pheno_small_scan.json

scan-harrington1d-pheno-small:
	python scripts/scan.py --config configs/harrington1d_pheno_small_scan.json

summarize-pheno-small:
	python scripts/summarize_scan.py \
		--input-dir results/scans/scala1d_pheno_small \
		--output-csv results/scans/scala1d_pheno_small_summary.csv
	python scripts/summarize_scan.py \
		--input-dir results/scans/harrington1d_pheno_small \
		--output-csv results/scans/harrington1d_pheno_small_summary.csv

plot-tf-vs-p:
	python scripts/plot_tf_vs_p.py \
		--scala-csv results/scans/scala1d_pheno_small_summary.csv \
		--harrington-csv results/scans/harrington1d_pheno_small_summary.csv \
		--output results/figures/tf_vs_p.png \
		--logx --logy

summarize-scala1d-pheno:
	python scripts/summarize_scan.py \
		--input-dir results/scans/scala1d_pheno \
		--output-csv results/scans/scala1d_pheno_summary.csv

summarize-scala1d-cc:
	python scripts/summarize_scan.py \
		--input-dir results/scans/scala1d_cc \
		--output-csv results/scans/scala1d_cc_summary.csv

summarize-harrington1d-pheno:
	python scripts/summarize_scan.py \
		--input-dir results/scans/harrington1d_pheno \
		--output-csv results/scans/harrington1d_pheno_summary.csv

summarize-harrington1d-cc:
	python scripts/summarize_scan.py \
		--input-dir results/scans/harrington1d_cc \
		--output-csv results/scans/harrington1d_cc_summary.csv

scan-harrington2d-cc:
	python scripts/scan.py --config configs/harrington2d_code_capacity_scan.json

scan-harrington2d-pheno:
	python scripts/scan.py --config configs/harrington2d_phenomenological_scan.json

summarize-harrington2d-cc:
	python scripts/summarize_scan.py \
		--input-dir results/scans/harrington2d_cc \
		--output-csv results/scans/harrington2d_cc_summary.csv

summarize-harrington2d-pheno:
	python scripts/summarize_scan.py \
		--input-dir results/scans/harrington2d_pheno \
		--output-csv results/scans/harrington2d_pheno_summary.csv

plot-harrington2d:
	python scripts/plot_harrington2d.py \
		--cc-csv results/scans/harrington2d_cc_summary.csv \
		--pheno-csv results/scans/harrington2d_pheno_summary.csv \
		--output-dir results/figures \
		--logx --logy-cc --logy-pheno

scan-scala2d-cc:
	python scripts/scan.py --config configs/scala2d_code_capacity_scan.json

scan-scala2d-pheno:
	python scripts/scan.py --config configs/scala2d_phenomenological_scan.json

summarize-scala2d-cc:
	python scripts/summarize_scan.py \
		--input-dir results/scans/scala2d_cc \
		--output-csv results/scans/scala2d_cc_summary.csv

summarize-scala2d-pheno:
	python scripts/summarize_scan.py \
		--input-dir results/scans/scala2d_pheno \
		--output-csv results/scans/scala2d_pheno_summary.csv

plot-scala2d:
	python scripts/plot_scala2d.py \
		--cc-csv results/scans/scala2d_cc_summary.csv \
		--pheno-csv results/scans/scala2d_pheno_summary.csv \
		--output-dir results/figures \
		--logx --logy-cc --logy-pheno

test-scala2d:
	python -m python.tests.test_scala2d_code_capacity
	python -m python.tests.test_scala2d_distance_trend
	python -m python.tests.test_scala2d_phenomenological

plot-scala1d-pheno-paper:
	python scripts/plot_scala1d_pheno_paper.py \
		--p-csv results/scans/scala1d_pheno_p_summary.csv \
		--q-csv results/scans/scala1d_pheno_q_summary.csv \
		--output results/figures/scala1d_pheno_paper.png

plot-scala2d-pheno-paper:
	python scripts/plot_scala2d_pheno_paper.py \
		--csv results/scans/scala2d_pheno_summary.csv \
		--output results/figures/scala2d_pheno_paper.png

plot-scala2d-cc-paper:
	python scripts/plot_scala2d_cc_paper.py \
		--csv results/scans/scala2d_cc_summary.csv \
		--output results/figures/scala2d_cc_paper.png

scan-scala2d-equal:
	python scripts/scan.py --config configs/scala2d_pheno_equal_scan.json

summarize-scala2d-equal:
	python scripts/summarize_scan.py \
		--input-dir results/scans/scala2d_pheno_equal \
		--output-csv results/scans/scala2d_pheno_equal_summary.csv

plot-scala2d-equal:
	python scripts/plot_scala2d_pheno_paper.py \
		--csv results/scans/scala2d_pheno_equal_summary.csv \
		--output results/figures/scala2d_pheno_paper.png

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PY_NATIVE_DIR)/harrington2d*.so
	rm -f $(PY_NATIVE_DIR)/harrington2d*.dylib
	rm -f $(PY_NATIVE_DIR)/scala2d*.so
	rm -f $(PY_NATIVE_DIR)/scala2d*.dylib
