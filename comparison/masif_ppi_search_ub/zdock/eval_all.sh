
#for ppi_pair in 3F74_A_B 3HRD_E_H 1ERN_A_B 2B3Z_C_D 4KGG_C_A 2FE8_A_C 1SOT_A_C 2LBU_E_D 3PGA_1_4 3QWN_I_J 3TND_B_D 1XDT_T_R
i=1
while read ppi_pair;
do
	echo "Starting $ppi_pair"
	if !((i % 5)); then
		./eval_one.sh $ppi_pair
	else
		./eval_one.sh $ppi_pair &
	fi
	((i++))


done < ../benchmark_list.txt
