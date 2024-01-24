Latest version (v10, 23 June 2023) contains several minor bug fixes. We thank Burhan Sarfraz (UCL) for catching the bugs!



- The dataset contains the United Nations General Debate Corpus (UNGDC) covering the period 1946-2022.
	"UNGDC_1946-2022.tgz"

- There are 10,568 speeches in plain text format (UTF8). Speeches are structured by Year (Session). Each speech is named using the following convention: ISO 3166-1 alpha-3 country code, followed by the UN Session number, followed by year. E.g. USA_75_2020.txt.

- The collection also contains a file (Speakers_by_session.xlsx) recording names and posts of speakers in UN General Debates. Note: before 1994 the UN records do not consistently identify the posts of all speakers.

- Original source files contain verbatim daily transcripts of the UN General Debate (and any other business on the UN agenda on the same day). Transcripts (in PDF format) were made available by the UN Library and processed to produce the UNGD Corpus as described in the paper. The original source files (PDFs) are in several tarballs: 
	“Raw_PDFs_1946-1969.tgz”; 
	“Raw_PDFs_1970-1990.tgz”; 
	“Raw_PDFs_1991-2022.tgz”.    

- When using the UNGDC data, please cite: 

Slava Jankin, Alexander Baturo, and Niheer Dasandi. "Words to Unite Nations: The Complete UN General Debate Corpus, 1946-Present." OSF working paper, https://osf.io/6kty4

AND 

Alexander Baturo, Niheer Dasandi, and Slava Mikhaylov, "Understanding State Preferences With Text As Data: Introducing the UN General Debate Corpus" Research & Politics, 2017.