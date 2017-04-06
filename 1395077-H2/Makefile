all:
	gcc datatrim.c -o datatrim
	gcc serialtester.c Lab4_IO.c -o serialtester -lm 
	mpicc -g -Wall main.c Lab4_IO.c -o main -lm

clean:
	rm datatrim
	rm serialtester
	rm data_input
	rm data_output
	rm main
