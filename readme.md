# Simple Example of the Pytorch-Lightning PredictionWriter

Usually, writing predictions is supported for the .predict() step, however it is also possible in .test(). 
This might be usefull if you want to safe the results of your test instances and later want to compute additional metrics or analyze results on your test set. 

Using the PredictionWriter is actually not that hard. You will only have to do some minor modifications: 
- Return relevant information in your lightningmodule's 'test_step()' function 
- Implement your custom PredictionWriter (or use the spartanic version that I created in 'prediction_writer.py')
    - To use it in .test(), be sure to overwrite the 'on_test_batch_end' or 'on_test_epoch_end' hooks 
- Pass the custom PredictionWriter to your trainer 'Trainer(..., callbacks=[custom_pred_writer])'