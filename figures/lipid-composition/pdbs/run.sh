# bash get_lipid_composition_popc.sh
# bash get_lipid_composition_pampa.sh
# bash get_lipid_composition_bbb.sh

# python3 center_density.py --input mass_POPC_popc_chol_dummy.out --z_center 104.545 --N 30 --output mass_POPC_popc_chol_dummy_centered.out
# python3 center_density.py --input mass_CHL1_popc_chol_dummy.out --z_center 104.545 --N 30 --output mass_CHL1_popc_chol_dummy_centered.out

# python3 center_density.py --input mass_DOPC_pampa_dummy.out --z_center 110.985 --N 30 --output mass_DOPC_pampa_dummy_centered.out
# python3 center_density.py --input mass_DOPS_pampa_dummy.out --z_center 110.985 --N 30 --output mass_DOPS_pampa_dummy_centered.out
# python3 center_density.py --input mass_CHL1_pampa_dummy.out --z_center 110.985 --N 30 --output mass_CHL1_pampa_dummy_centered.out

# python3 center_density.py --input mass_POPC_bbb_dummy.out --z_center 106.075 --N 30 --output mass_POPC_bbb_dummy_centered.out
# python3 center_density.py --input mass_CHL1_bbb_dummy.out --z_center 106.075 --N 30 --output mass_CHL1_bbb_dummy_centered.out
# python3 center_density.py --input mass_SAPC_bbb_dummy.out --z_center 106.075 --N 30 --output mass_SAPC_bbb_dummy_centered.out
# python3 center_density.py --input mass_SAPE_bbb_dummy.out --z_center 106.075 --N 30 --output mass_SAPE_bbb_dummy_centered.out
# python3 center_density.py --input mass_SAPI_bbb_dummy.out --z_center 106.075 --N 30 --output mass_SAPI_bbb_dummy_centered.out
# python3 center_density.py --input mass_SAPS_bbb_dummy.out --z_center 106.075 --N 30 --output mass_SAPS_bbb_dummy_centered.out
# python3 center_density.py --input mass_SLPC_bbb_dummy.out --z_center 106.075 --N 30 --output mass_SLPC_bbb_dummy_centered.out
# python3 center_density.py --input mass_SOPE_bbb_dummy.out --z_center 106.075 --N 30 --output mass_SOPE_bbb_dummy_centered.out
# python3 center_density.py --input mass_OSM_bbb_dummy.out --z_center 106.075 --N 30 --output mass_OSM_bbb_dummy_centered.out


python3 visualize.py --set 3 --set_name N --files mass_{POPC,SAPC,SAPE,SAPS,SLPC,SOPE,OSM}_bbb_dummy_centered.out --outfile bbb_N.png
python3 visualize.py --set 2 --set_name P --files mass_{POPC,SAPC,SAPE,SAPI,SAPS,SLPC,SOPE,OSM}_bbb_dummy_centered.out --outfile bbb_P.png
python3 visualize.py --set 1 --set_name C2-18 --files mass_{POPC,SAPC,SAPE,SAPI,SAPS,SLPC,SOPE,OSM}_bbb_dummy_centered.out --outfile bbb_c2-18.png
python3 visualize.py --set 1 2 --set_name C26 O3 --files mass_CHL1_bbb_dummy_centered.out --outfile bbb_chol.png

python3 visualize.py --set 1 2 --set_name C26 O3 --files mass_CHL1_pampa_dummy_centered.out --outfile pampa_chol.png
python3 visualize.py --set 2 --set_name P --files mass_{DOPC,DOPS}_pampa_dummy_centered.out --outfile pampa_P.png
python3 visualize.py --set 3 --set_name N --files mass_{DOPC,DOPS}_pampa_dummy_centered.out --outfile pampa_N.png
python3 visualize.py --set 1 --set_name C2-18 --files mass_{DOPC,DOPS}_pampa_dummy_centered.out --outfile pampa_c2-18.png

python3 visualize.py --set 1 2 3 --set_name C2-18 P N --files mass_POPC_popc_chol_dummy_centered.out --outfile popc_popc.png
python3 visualize.py --set 1 --set_name C2-18 --files mass_POPC_popc_chol_dummy_centered.out --outfile popc_c2-18.png
python3 visualize.py --set 2 --set_name P --files mass_POPC_popc_chol_dummy_centered.out --outfile popc_P.png
python3 visualize.py --set 3 --set_name N --files mass_POPC_popc_chol_dummy_centered.out --outfile popc_N.png
python3 visualize.py --set 1 2 --set_name C26 O3 --files mass_CHL1_popc_chol_dummy_centered.out --outfile popc_chol.png
