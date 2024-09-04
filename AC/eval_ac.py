from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
import os
# from huggingface_hub import login
import json
import argparse
from arithmeticcoding import *

def tensor_to_numpy(tensor):
    # Ensure the tensor is on CPU and convert it to NumPy
    conv = np.array(tensor.cpu().numpy(), dtype=np.uint32, order='C')
    # ensure that eaach value is uint32
    for i in range(len(conv)):
        conv[i] = np.uint32(conv[i])
    return conv

def numpy_to_tensor(array: np.ndarray, device):
    # Convert the NumPy array back to a PyTorch tensor
    # convert the type of the np array to float32
    array = array.astype(np.float32)
    return torch.tensor(array, device=device)

def text_to_tokens(tokenizer, text):
    # ignore the warning that this gives about too many tokens
    tokens = tokenizer(text, return_tensors="pt")
    tokens = tokens["input_ids"].squeeze()
    return tokens

def read_bitstream(bitin):
    temp_list = []
    while True:
        temp = bitin.read()
        if temp == -1:
            break
        temp_list += [temp]
    temp_arr = np.array(temp_list)
    final_ind = (np.where(temp_arr==1)[0][-1]).astype(int)
    final_arr = temp_arr[:final_ind+1]
    
    return final_arr

class AC_Encode:
    def __init__(self, model, tokenizer, compressed_file_name, output_dir, device, batch_size, window_size):
        self.model = model
        self.tokenizer = tokenizer
        self.compressed_file_name = compressed_file_name
        self.batch_size = batch_size
        self.window_size = window_size
        self.AC_files = []
        self.AC_bitouts = []
        self.AC_encoders = []
        self.output_dir = output_dir

        for i in range(batch_size):
            AC_file_name = f"{output_dir}/{i}_AC.txt"
            file_out = open(AC_file_name, 'wb')
            bitout = BitOutputStream(file_out)
            AC_encoder = ArithmeticEncoder(32, bitout)
            self.AC_files.append(file_out)
            self.AC_bitouts.append(bitout)
            self.AC_encoders.append(AC_encoder)
        self.device = device

    def pad(self, tokens, padding_val):
        pad_len = self.batch_size - len(tokens) % self.batch_size
        if pad_len != self.batch_size:
            padding = torch.tensor([padding_val]*pad_len)

            tokens = torch.cat((tokens, padding))

        else:
            pad_len = 0

        return tokens, pad_len

    def encode_from_tokens(self, tokens_full):
        tokens_full, pad_len = self.pad(tokens_full, self.tokenizer.eos_token_id)
        tokens = tokens_full.view(self.batch_size, -1)

        eos = torch.tensor([self.tokenizer.eos_token_id]*tokens.shape[0]).unsqueeze(1)
        tokens = torch.cat((eos, tokens), 1)

        length = tokens.shape[1]
        for b_ind in tqdm(range(1, length)):
            vocab_size = self.encode_batch(tokens[:, max(0, b_ind - self.window_size):b_ind], tokens[:, b_ind])


        for encoder in self.AC_encoders:
            encoder.finish()

        for bitout in self.AC_bitouts:
            bitout.close()

        for f in self.AC_files:
            f.close()
       
        return tokens.shape[1] - 1, pad_len, vocab_size

        
    @torch.no_grad()
    def encode_batch(self, prompt_tokens, target):
        prompt_tokens = prompt_tokens.to(self.device)
        if len(prompt_tokens.shape) == 1:
            prompt_tokens = prompt_tokens.unsqueeze(1)
        my_inputs = {}
        my_inputs['input_ids'] = prompt_tokens
        output = self.model(**my_inputs)
        logits = output.logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)
        probs_np2 = probs.cpu().numpy()
        cumul = np.zeros(probs.shape[-1] + 1, dtype = np.uint64)
        
        for j in range(self.batch_size):
            prob1 = probs_np2[j]
            cumul[1:] = np.cumsum(prob1*10000000 + 1)
            self.AC_encoders[j].write(cumul, target[j])
        return probs.shape[-1]

    def compute_compression_ratio(self,tokens_encoded, time, text_file):
        text_encoded = self.tokenizer.decode(tokens_encoded.squeeze().tolist())
        
        N_T = len(tokens_encoded)
        N_C = len(text_encoded)
        
        df_out = {}
        df_out['$N_C$'] = [N_C]
        df_out['$N_T$'] = [N_T]
        df_out['$T'] = time
        
        compressed_bits_count = 0
        compressed_size = 0
        dataset_size = os.path.getsize(text_file)
        for i in range(self.batch_size):
            AC_file_name = f"{self.output_dir}/{i}_AC.txt"
            compressed_size += os.path.getsize(AC_file_name)
            file_in = open(AC_file_name, 'rb')
            bitin = BitInputStream(file_in)
            compressed_bits = read_bitstream(bitin)
            compressed_bits_count += compressed_bits.size
            file_in.close()
        
            
        compression_ratios = compressed_size/dataset_size
        rho_AC = compressed_bits_count/N_C
        print(f'Compression Ratio for Arithmetic Coding :  {rho_AC} bits/char')
       
        df_out['AC compressed file size'] = [compressed_bits_count]
        df_out['$\rho_{LLaMa+AC}$'] = [rho_AC]
        df_out["Compression Ratio"] = [compression_ratios]
            
        with open(f"{self.output_dir}/metrics.json", 'w') as file_metrics: 
            json.dump(df_out, file_metrics)
            
class AC_Decode:
    def __init__(self, model, tokenizer, device, output_dir, batch_size, window_size):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.AC_files = []
        self.AC_bitins = []
        self.AC_decoders = []
        self.batch_size = batch_size
        self.window_size = window_size

        for i in range(batch_size):
            AC_file_name = f"{output_dir}/{i}_AC.txt"
            file_in = open(AC_file_name, 'rb')
            bitin = BitInputStream(file_in)
            AC_decoder = ArithmeticDecoder(32, bitin)
            self.AC_files.append(file_in)
            self.AC_bitins.append(bitin)
            self.AC_decoders.append(AC_decoder)

    @torch.no_grad()
    def decode_AC(self, total_length, pad_len, vocab_size):
        cumul = np.zeros(vocab_size +1, dtype = np.uint64)

        tokens = torch.tensor([self.tokenizer.eos_token_id]*self.batch_size).unsqueeze(1)
        tokens = tokens.to(self.device)
        
        index = 0
        print("Reading")
        while index < total_length:
            
            input = tokens[:, max(0, index-self.window_size + 1): index + 1]
            my_inputs = {}

            my_inputs['input_ids'] = input
            outputs = self.model(**my_inputs) 
            logits = outputs.logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1) 

            probs_np = probs.cpu().numpy()

            next_tokens = torch.zeros((self.batch_size, 1), dtype=int).to(self.device)
            for j in range(self.batch_size):
                cumul[1:] = np.cumsum(probs_np[j]*10000000 + 1)
                next_tokens[j] = self.AC_decoders[j].read(cumul, probs_np[j].size)

            tokens = torch.cat((tokens, next_tokens), dim=1)
            index += 1
        
        tokens = tokens[:, 1:].int()
        tokens = tokens.flatten()
        if pad_len != 0:
            tokens = tokens[:-pad_len]
        
        text = self.tokenizer.batch_decode(tokens)
        text = "".join(text)

        for bitin in self.AC_bitins:
            bitin.close()

        for f in self.AC_files:
            f.close()
        
        return text   

def verify_text(compressed_file_name,text_file, text_decoded):
    with open(text_file,'r') as txt_enc:
        text_encoded = txt_enc.read()


    if text_decoded[:17] == "<|begin_of_text|>":
        text_decoded = text_decoded[17:]

    if text_encoded == text_decoded:
        print(f'Successful decoded')
    else:
        print("********!!!!! Error !!!!!*********")


    with open(compressed_file_name+'_AC_decoded_text.txt','w') as txt_dec:
        txt_dec.write(text_decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--tokenizer",
        type=str
    )
    parser.add_argument(
        '--batch_size',
        type=int
    )
    parser.add_argument(
        '--context_size',
        type=int
    )
    parser.add_argument(
        '--input_file',
        type=str
    )
    parser.add_argument(
        '--AC_output_dir',
        type=str
    )
    parser.add_argument(
        '--encode_decode',
        default= 0,
        type=str
    )
    

    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model.eval()
    device = torch.cuda.current_device()
    model.to(device)
    compressed_file_name = args.input_file[:-4]
    with open(args.input_file,'r') as f_in:
            text_input = f_in.read()
    os.makedirs(args.AC_output_dir,exist_ok=True)
    Encoder = AC_Encode(model, tokenizer, compressed_file_name, args.AC_output_dir, device, args.batch_size, args.context_size)
    
    tokens_full = text_to_tokens(tokenizer, text_input)
    
    start = time.time()
    total_length, pad_len, vocab_size = Encoder.encode_from_tokens(tokens_full)
    end = time.time()
    
    compression_time = end - start
    Encoder.compute_compression_ratio(tokens_full, compression_time, args.input_file)

    if args.encode_decode == "1":
        Decoder = AC_Decode(model, tokenizer, device, args.AC_output_dir, args.batch_size, args.context_size)
        decoded_text_ac = Decoder.decode_AC(total_length, pad_len, vocab_size)
        verify_text(compressed_file_name, args.input_file, decoded_text_ac)

if __name__ == "__main__":
    main()



