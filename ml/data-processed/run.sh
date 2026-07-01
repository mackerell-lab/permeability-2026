for dataset in  # bbb # pampa bbb popc_chol popc_chol_halogen
do
	for type in biogen # extended
	do
		for property in P_app ER # 
		do

			mkdir -p ${dataset}-${type}/${property}
			cd ${dataset}-${type}/${property}

			for filetype in lgfe # ie lgfe_ie lgfe_ie_sym_avg lgfe_ie_mw # 
			do
				# nohup python3 ../../../scripts/data_curation.py ../../../data/${dataset}-${type}-${property}_${filetype}.csv 5 & >> ../../history.txt
				nohup python3 ../../../scripts/data_curation.py ../../../data/${dataset}-${type}-${property}-including-protomers_${filetype}.csv 5 & >> ../../history.txt
			done

			cd ../../
		done
	done
done
