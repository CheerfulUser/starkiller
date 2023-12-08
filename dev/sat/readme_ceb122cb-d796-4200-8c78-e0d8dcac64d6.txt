This file contains an overview of the structure and content of your download request from the ESO Science Archive.


For every downloaded dataset, files are listed below with the following structure:

dataset_name
        - archive_file_name (technical name, as saved on disk)	original_file_name (user name, contains relevant information) category size


Please note that, depending on your operating system and method of download, at download time the colons (:) in the archive_file_name as listed below may be replaced by underscores (_).


In order to rename the files on disk from the technical archive_file_name to the more meaningful original_file_name, run the following shell command:
    cat THIS_FILE | awk '$2 ~ /^ADP/ {print "test -f",$2,"&& mv",$2,$3}' | sh


In case you have requested cutouts, the file name on disk contains the TARGET name that you have provided as input. To order files by it when listing them, run the following shell command:
    cat THIS_FILE | awk '$2 ~ /^ADP/ {print $2}' | sort -t_ -k3,3


Your feedback regarding the data quality of the downloaded data products is greatly appreciated. Please contact the ESO Archive Science Group via https://support.eso.org/ , subject: Phase 3 ... thanks!

The download includes contributions from the following collections:
Ref(0)	MUSE-DEEP	https://doi.eso.org/10.18727/archive/42	IDP_MUSE_DEEP_release_description_1.5.2.pdf	https://www.eso.org/rm/api/v1/public/releaseDescriptions/102
Ref(1)	MUSE	https://doi.eso.org/10.18727/archive/41	IDP_MUSE_IFU_release_description_1.8.2.pdf	https://www.eso.org/rm/api/v1/public/releaseDescriptions/78

Publications based on observations collected at ESO telescopes must acknowledge this fact (please see: http://archive.eso.org/cms/eso-data-access-policy.html#acknowledgement). In particular, please include a reference to the corresponding DOI(s). They are listed in the third column in the table above and referenced below for each dataset. The following shell command lists them:

	cat THIS_FILE | awk -F/ '$1 ~ /^Ref\(/ {print $0,$NF}' | awk '{print $2, $3}' | sort | uniq


Each collection is described in detail in the corresponding Release Description. They can be downloaded with the following shell command:

	cat THIS_FILE | awk -F/ '$1 ~ /^Ref\(/ {print $0,$NF}' | awk '{printf("curl -o %s_%s %s\n", $6, $4, $5)}' | sh

ADP.2023-03-10T08:28:07.890 Ref(0)
	- ADP.2023-03-10T08:28:07.890.fits	MU_SCBD_3250018_2022-06-24T08:46:05.083_WFM-NOAO-N_OBJ.fits	SCIENCE.CUBE.IFS	3304667520
	- ADP.2023-03-10T08:28:07.891.fits	MU_SIMD_3250018_2022-06-24T08:46:05.083_WFM-NOAO-N_OBJ.fits	ANCILLARY.IMAGE.WHITELIGHT	561600
	- ADP.2023-03-10T08:28:07.894.fits	MU_SXPM_3250018_2022-06-24T08:46:05.083_WFM-NOAO-N_OBJ.fits	ANCILLARY.EXPMAP	558720
ADP.2022-07-14T15:34:49.531 Ref(1)
	- ADP.2022-07-14T15:34:49.531.fits	MU_SCBC_3250018_2022-06-24T08:46:05.083_WFM-NOAO-N_OBJ.fits	SCIENCE.CUBE.IFS	3024843840
	- ADP.2022-07-14T15:34:49.532.fits	MU_SIMC_3250018_2022-06-24T08:46:05.083_WFM-NOAO-N_OBJ.fits	ANCILLARY.IMAGE.WHITELIGHT	518400
ADP.2022-08-10T13:11:28.614 Ref(1)
	- ADP.2022-08-10T13:11:28.614.fits	MU_SCBC_3250064_2022-07-23T08:58:30.223_WFM-NOAO-N_OBJ.fits	SCIENCE.CUBE.IFS	3015492480
	- ADP.2022-08-10T13:11:28.615.fits	MU_SIMC_3250064_2022-07-23T08:58:30.223_WFM-NOAO-N_OBJ.fits	ANCILLARY.IMAGE.WHITELIGHT	532800
