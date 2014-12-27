BINARY=volume
REPORT=report.txt
TESTFILE=test_models/starfighter2.stl
VERSIONS=simple_aos_global_seqsum simple_soa_global_seqsum simple_soa_shared_seqsum unroll_soa_global_parsum unroll_soa_global_seqsum unroll_soa_shared_parsum unroll_soa_shared_seqsum
THREADS=32 64 128 256 512 1024

all:
	for i in ${VERSIONS}; do \
		nvcc $$i/volume.cu $$i/main.cu -o $$i/${BINARY}; \
	done

test: all
	echo "GMU test with ${TESTFILE}" > ${REPORT}
	echo "" >> ${REPORT}
	echo "" >> ${REPORT}

	for t in ${THREADS}; do \
		echo "threads: $$t" >> ${REPORT}; \
		echo "" >> ${REPORT}; \
		for v in ${VERSIONS}; do \
			echo $$v >> ${REPORT}; \
			./$$v/${BINARY} $$t ${TESTFILE} >> ${REPORT}; \
			echo "" >> ${REPORT}; \
		done; \
		echo "" >> ${REPORT}; \
	done

clean:
	rm -f */${BINARY} ${REPORT}
