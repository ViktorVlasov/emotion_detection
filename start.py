import src

# split dataset
cfg = src.combine_config(cfg_path='src/configs/params.yaml')

if __name__ == "__main__":
    src.split_dataset(input_path=cfg.DATASET_EMO.PATH,
                      output_path=cfg.DATASET_EMO.SPLIT_OUTPUT_PATH,
                      test_size=cfg.DATASET_EMO.TEST_SIZE,
                      val_size=cfg.DATASET_EMO.VAL_SIZE,
                      balance_classes=cfg.DATASET_EMO.BALANCE_CLASSES)
    
