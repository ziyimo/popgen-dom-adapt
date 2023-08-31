GENERAL-INFO-START

	seq-file            deminf_seq.txt
	trace-file          deminf_mcmc.log				
	locus-mut-rate          CONST

	mcmc-iterations	  100000
	iterations-per-log  50
	logs-per-line       10

	find-finetunes		TRUE
#	finetune-coal-time	0.01		
#	finetune-mig-time	0.3		
#	finetune-theta		0.04
#	finetune-mig-rate	0.02
#	finetune-tau		0.0000008
#	finetune-mixing		0.003
#   finetune-locus-rate 0.3
	
	tau-theta-print		100000000
	tau-theta-alpha		1.0			# for STD/mean ratio of 100%
	tau-theta-beta		10000.0		# for mean of 1e-4

	mig-rate-print		0.000001    # *1e-6
	mig-rate-alpha		0.002
	mig-rate-beta		0.00001

GENERAL-INFO-END

CURRENT-POPS-START

	POP-START
		name	A
		samples	pop0samp0 h pop0samp1 h pop0samp2 h pop0samp3 h pop0samp4 h pop0samp5 h pop0samp6 h pop0samp7 h 
	POP-END

	POP-START
		name	B
		samples	pop1samp8 h pop1samp9 h pop1samp10 h pop1samp11 h pop1samp12 h pop1samp13 h pop1samp14 h pop1samp15 h pop1samp16 h pop1samp17 h pop1samp18 h pop1samp19 h pop1samp20 h pop1samp21 h pop1samp22 h pop1samp23 h pop1samp24 h pop1samp25 h pop1samp26 h pop1samp27 h pop1samp28 h pop1samp29 h pop1samp30 h pop1samp31 h pop1samp32 h pop1samp33 h pop1samp34 h pop1samp35 h pop1samp36 h pop1samp37 h pop1samp38 h pop1samp39 h 
	POP-END

	POP-START
		name	C
		samples	pop2samp40 h pop2samp41 h pop2samp42 h pop2samp43 h pop2samp44 h pop2samp45 h pop2samp46 h pop2samp47 h pop2samp48 h pop2samp49 h pop2samp50 h pop2samp51 h pop2samp52 h pop2samp53 h pop2samp54 h pop2samp55 h pop2samp56 h pop2samp57 h pop2samp58 h pop2samp59 h pop2samp60 h pop2samp61 h pop2samp62 h pop2samp63 h pop2samp64 h pop2samp65 h pop2samp66 h pop2samp67 h pop2samp68 h pop2samp69 h pop2samp70 h pop2samp71 h 
	POP-END

CURRENT-POPS-END

ANCESTRAL-POPS-START

	POP-START
		name			BC
		children		B		C
	POP-END

	POP-START
		name			ABC
		children		A		BC
	POP-END

ANCESTRAL-POPS-END

MIG-BANDS-START	

	BAND-START		
       source  B
       target  C
	BAND-END

	BAND-START		
       source  C
       target  B
	BAND-END

MIG-BANDS-END
