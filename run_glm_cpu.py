from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

model_path = "/mnt/data/chatglm3-6b"

# 定义10个测试问题
questions = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道",
    "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上\"意思\"分别是什么意思。",
    "小明今年比小红大3岁，5年前小明比小红大几岁？10年后小明比小红大几岁？",
    "请解释：'我看见了一个人在河边用望远镜' 这句话有几种理解方式？",
    "如果昨天是明天的话，那么今天就是星期五。那么今天实际上是星期几？",
    "张三站在李四的左边，王五站在李四的右边，赵六站在张三的左边。请问从左到右的顺序是什么？",
    "如果所有的猫都怕水，而Tom是一只猫，那么Tom怕水吗？但是如果Tom是一只会游泳的猫呢？"
]

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True 
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    local_files_only=True
).eval()

# 输出模型信息
print("=" * 80)
print(f"使用大模型：ChatGLM3-6B")
print(f"模型路径：{model_path}")
print("=" * 80)

# 循环处理10个问题
for i, prompt in enumerate(questions, 1):
    print(f"\n问题 {i}/10：")
    print(f"问题内容：{prompt}")
    print("-" * 60)
    print("模型回答：")
    
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    outputs = model.generate(
        inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=False
    )
    
    print("-" * 60)
    print(f"问题 {i} 回答完成")
    print("=" * 80)

print("\n所有问题处理完成！")