# Convert the fragments to their latents
for i in $(seq -f "%02g" 1 24)
do
  echo "Speaker $i"
  python custom_projector.py --ckpt=cartoon_stylegan2/networks/ffhq256.pt --e_ckpt=cartoon_stylegan2/networks/encoder_ffhq.pt  --file=stills/speaker$i.png --project_name=speaker$i
done

# Create an overview
python transform_images.py --out_dir=images --latents_dir=latents --models NaverWebtoon NaverWebtoon_StructureLoss NaverWebtoon_FreezeSG Romance101 TrueBeauty Disney Disney_StructureLoss Disney_FreezeSG Metface_StructureLoss Metface_FreezeSG --create_overview
python transform_images.py --out_dir=images --latents_dir=latents --models NaverWebtoon NaverWebtoon_StructureLoss NaverWebtoon_FreezeSG Romance101 TrueBeauty Disney Disney_StructureLoss Disney_FreezeSG Metface_StructureLoss Metface_FreezeSG