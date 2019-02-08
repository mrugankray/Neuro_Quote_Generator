# Neuro_Quote_Generator

->The original quote file can be found in downloaded directory.

->create_new_text_file.py has the necessary code to generate quotes from preprocessed quotes and the new quote file can be found in modified_text folder.

->char_lvl_rnn.py file will train the model to generate new quotes.

->AI generated quotes can be found in AI_generated_quotes folder.

->I have trained my model for 20 epochs but you can change the value of n_epochs in line 290 
i.e n_epochs = 20 "start smaller if you are just testing initial behavior".

->you can play around batch_size as well i have used 128 it can be found at line 288
i.e batch_size = 128 

->seq_length is like the the stride in cnn, you can change it according to your requirements,it can be found at line 289
i.e seq_length = 100
