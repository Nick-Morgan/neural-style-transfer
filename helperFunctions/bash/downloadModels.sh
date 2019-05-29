: '
Purpose:
	If any of the listed files are not in the models folder, download them
	via wget. The .ckpt files originated from Logan Engstrom. The VGG19
	model originated from http://www.vlfeat.org. All files were
	moved to my Google Drive to ensure availability during my
	presentation
'
base_url="https://drive.google.com/uc?export=download&id="

declare -a ids=(
	1IVmO4u6KL-vDZRFturgc87A6HOGtqSBV
	1Go_JSJcJ9rAJW7FMkcmodPYYO4Br-Sg6
	118qvV0EoW0HovExfS4w3EAx30ynmGjJr
	1IQF3nIMEb0CUrbeKFs_JD73gHcI8wJ_Z
	12NRMtaUJSJbe42-J6iUutDn_BW2lgaTm
	12idcbTtlF8Ms2uauM4dQtDAP-oQ3bkB6
	1CeOmI8FtTZyJhPQtCkQ8occtFOwfyhNY)

declare -a names=(
	imagenet-vgg-verydeep-19.mat
	scream.ckpt
	rain_princess.ckpt
	la_muse.ckpt
	wreck.ckpt
	wave.ckpt
	udnie.ckpt)

for idx in "${!ids[@]}"
do
	id="${ids[$idx]}"
	name="${names[$idx]}"
	
	if ! [ -e models/"$name" ]; then
		
		# imagenet model is large, and thus needs additional wget logic to accept
		# google's large file redirect
		if [ "$name" = "imagenet-vgg-verydeep-19.mat" ]; then
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$id" -O "models/$name" && rm -rf /tmp/cookies.txt

		else # every other file can just use simple wget
			wget "$base_url$id" -O "models/$name"
		fi
	fi
done
