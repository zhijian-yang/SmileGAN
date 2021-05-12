#!/bin/bash

while read requirement;
do 
	if conda install --yes $requirement; then
		echo "Successfully install: ${requirement}"
	else
		conda install --yes -c conda-forge $requirement
	fi
done < requirements.txt
