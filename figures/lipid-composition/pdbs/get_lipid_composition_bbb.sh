iter=bbb_dummy

# Gives out the upper z bound of the box
High=$(bounding $iter".pdb" $iter".pdb" "all" | grep "Bounds" | awk '{print $4;}' | sed 's/^.*,//g' |  cut -d")" -f1)

# Gives out the lower z bound of the box
Low=$(bounding $iter".pdb" $iter".pdb" "all" | grep "Bounds" | awk '{print $2;}' | sed 's/^.*,//g' |  cut -d")" -f1)

# To make it to nearest lower integer, removing the decimals But in 
# the case of negative numbers this won't work. Thus, delibrately
# deducting 1. 

minz=${Low%.*}

if [[ $minz -lt 0 ]] ; then
minz=$((minz-1))
fi
# To make it to nearest higher integer, removing decimals and adding 1
# Same logic apply for maxz for negative members as above

maxz=${High%.*}

if [[ $maxz -ge 0 ]] ; then
maxz=$((maxz+1))
fi

nbins=$((maxz-minz))

for arg in POPC CHL1 OSM SAPC SAPE SAPI SAPS SLPC SOPE
do
if [[ $arg == "CHL1" ]]; then

    density-dist --type=mass -- $iter".pdb" $iter".pdb" $minz $maxz $nbins "resname == '$arg' && name == 'C26'" "resname == '$arg' && name == 'O3'" > "mass_"$arg"_"$iter".out"

elif [[ $arg == "OSM" ]]; then
    density-dist --type=mass -- $iter".pdb" $iter".pdb" $minz $maxz $nbins "resname == '$arg' && name == 'C18F'" "resname == '$arg' && name == 'P'" "resname == '$arg' && name == 'N'" > "mass_"$arg"_"$iter".out"

else
    density-dist --type=mass -- $iter".pdb" $iter".pdb" $minz $maxz $nbins "resname == '$arg' && name == 'C218'" "resname == '$arg' && name == 'P'" "resname == '$arg' && name == 'N'" > "mass_"$arg"_"$iter".out"
fi

done
echo "$iter done"

echo " For $iter "

echo " Minimum Z $minz "
echo " Maximum Z $maxz "
echo " Np. of bins $nbins"
