# create metadata for hf dataset

splits=(test-clean test-other dev-clean dev-other train-clean-100 train-clean-360 train-other-500)

for split in "${splits[@]}"; do
  cd $DATA/LibriSpeech/$split
  cat */*/*.txt | cut -d ' ' -f1 | cut -d'-' -f1,2 | sed 's;-;/;g' > file_root.tmp
  cat */*/*.txt | cut -d ' ' -f1 | paste -d / file_root.tmp - > file_names.tmp
  sed -i 's/$/.flac/' file_names.tmp
  cat */*/*.txt | cut -d ' ' -f2- | paste -d , file_names.tmp - > file_names_and_trans.tmp
  sed -i "s|$|,${split}|" file_names_and_trans.tmp
  echo "file_name,text,split" > metadata.csv
  cat file_names_and_trans.tmp >> metadata.csv

  cat file_names_and_trans.tmp
  rm *.tmp
done

cd $EXP/flow_matching_speech/examples/text/
