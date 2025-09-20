schema = {
  "required": ["image_id","modality","crs","pixel_spacing","tile_size","bounds","provenance","label_set"]
}

def test_record_required_fields():
    rec = {
      "image_id": "S1_20210111_T1234_tile_00042",
      "modality": "sentinel1_sar_vv",
      "crs": "EPSG:32644",
      "pixel_spacing": [10.0,10.0],
      "tile_size": [512,512],
      "bounds": [0,0,1,1],
      "provenance": {"source_uri": "s3://x", "processing": ["speckle_filter:lee","normalize:0-1"]},
      "label_set": ["flooded","non_flooded"]
    }
    for k in schema['required']:
        assert k in rec
