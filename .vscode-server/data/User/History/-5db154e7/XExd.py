import csv

input_csv = "/home/tqsang/x2_lab/abnormal.csv"
output_txt = "abnorm_generated_prompts.txt"

negative_prompt = 'EasyNegative, bad-hands-5, (((nude, naked,child, child face))), un-detailed skin, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, ugly eyes, (out of frame:1.3), worst quality, low quality, jpeg artifacts, cgi, sketch, cartoon, drawing, (out of frame:1.1), worst quality, low quality, jpeg artifacts, poorly drawn, (((word, words, letter, text, signature, watermark)))'
steps = 80
width = 540
height = 540

def process_row(row):
    return " ".join(row[1:])

with open(input_csv, newline='') as csvfile:
    reader = csv.reader(csvfile)
    with open(output_txt, "w") as txtfile:
        for row in reader:
            prompt = process_row(row)
            cmd = f'--prompt "{prompt}" --negative_prompt "{negative_prompt}" --steps {steps} --sampler_name "DDIM" --width {width} --height {height}\n'
            txtfile.write(cmd)
