embs_dir="$LAION_ICONS_DIR/embs"
meta_dir="$LAION_ICONS_DIR/meta"
mkdir -p "$embs_dir"
mkdir -p "$meta_dir"
for shard in {0..100}; do
    shard=$(printf '%04d' $shard)
    echo "Working on shard $shard ..."

    filename="metadata_$shard.parquet"
    curl -o "$meta_dir/$filename" https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/laion2B-en-metadata/$filename

    filename="img_emb_$shard.npy"
    curl -o "$embs_dir/$filename" https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/$filename
done
