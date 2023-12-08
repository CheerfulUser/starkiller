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
Ref(0)	MUSE	https://doi.eso.org/10.18727/archive/41	IDP_MUSE_IFU_release_description_1.8.2.pdf	https://www.eso.org/rm/api/v1/public/releaseDescriptions/78

Publications based on observations collected at ESO telescopes must acknowledge this fact (please see: http://archive.eso.org/cms/eso-data-access-policy.html#acknowledgement). In particular, please include a reference to the corresponding DOI(s). They are listed in the third column in the table above and referenced below for each dataset. The following shell command lists them:

	cat THIS_FILE | awk -F/ '$1 ~ /^Ref\(/ {print $0,$NF}' | awk '{print $2, $3}' | sort | uniq


Each collection is described in detail in the corresponding Release Description. They can be downloaded with the following shell command:

	cat THIS_FILE | awk -F/ '$1 ~ /^Ref\(/ {print $0,$NF}' | awk '{printf("curl -o %s_%s %s\n", $6, $4, $5)}' | sh

ADP.2022-07-14T15:54:47.933 Ref(0)
	- ADP.2022-07-14T15:54:47.933.fits	MU_SCBX_3236622_2022-06-28T10:00:42.468_WFM-NOAO-E_SKY.fits	SCIENCE.CUBE.IFS	2931929280
	- ADP.2022-07-14T15:54:47.934.fits	MU_SIMX_3236622_2022-06-28T10:00:42.468_WFM-NOAO-E_SKY.fits	ANCILLARY.IMAGE.WHITELIGHT	504000
