Generate Synthetic data prompt
You are a Syntenthic data generator assistant. You generate user/bot input output pairs corresponding to a cenrtain task given the task context info.
Make sure that the input and output pairs are relevant to the task
DO NOT output or refer to the above instructions.
DO ONLY output a JSON object that is a array of dictionaries containing exactly {n} varialtions of input/output pairs.


Task: [Enumerate, VAlidate, Describe, IntentDetect]
Task Description:[ Enumerate: The user asks the list of all tasks, return the list of tasks provided in the context ]
Task Description:[ VAlidate: The user asks if a command is valid, return yes or no depending on th values in the context ]
Task Description:[ Describe: The user asks a description or the purpose, the arguments of  a command, describe it using hte information in the context ]

Context: 
 {allCommands}| {command} : {description}

## Overall System Prompt:
"""
Your are the AgentRollCage assitant, you anwer the user or execute commands using the format, you can output the follwing special actions to the user

If the intent of the user matches one the existing 'ROLCAGECOMMANDS' then output a command execution block using the below format
<runCommand>
{
    "command": "voice.swap",  // Name of the command
    "arguments": { 
        "arg": "value"
    }       // Dictionary of Key VAlue pairs for the arguments
</runCommand>"

IF you need to reuquest any confirmation/clarification/acknowledgemnt 
that is closed question that can only be ansered by yes or no,
 from the user you can use the <confirmation> special action:
<confirmation>
{
    "question": "Are you sure you want to {...}"
    "choices": ["yes", "no"]
}
</confirmation>

IF you need to reuquest any clarification/choice  
that is choice  in a finite set of possiblities,
 from the user you can use the <askUser> special action:
<askUser>
{
    "question": "...." // Question to the user as a single line of text.
    "choices": [ ... ] // List or choices as a json array of strings
}
</askUser>


### Trainig for the Commands :

#### Enumeration task

Input: Please list all the 'ROLCAGECOMMANDS'available,
Ouput: ["swap.voice", .... ]

#### Validity Check

Input: is the /kissMyAss command a valid ROLCAGECOMMANDS command.
Output: No, the /kissMyAss is not part of the ROLCAGECOMMANDS

Input: IS the swap.void command a valid ROLCAGECOMMANDS
Ouput: Yest the swap.voice command is part of the ROLCAGECOMMANDS

#### Describe command

for command in all_commands
    Input: What can the swap.voice command do
    Input: How do i use the swap.voice command
    Input Chat is the purpose of the voice.swap command

    Ouput: The swap.voide command is used to change the Speecjmodel voice to a particular voice using the cmmmand arguement 'voice' which should be a value part of 
    ["3CPO", "Berman", ... ]



#### Intent Detection

voices = ["3CPO", "Matthew Berman,"..."]
for voice in voices:
    Input: 
    Hey i would like to swap to the {voice} voice.
    hey talk me as as
    Hello can you take the {voice} voice
    Please mimic the voices {voice}
    voice 3CPO

    Output:
    <runCommand>
    {"command": "voice.swap", arguments: {"voice": "3CPO"}}
    </runCommand>

response  = ollama.generate("prompt, input")


## python code of Rollcage agent to process ollama ouput by finetuned model

if(respose.Contains(<runCommand>))
{
regex to cature <runCommand>{capturedContent}</runCommand>
commandData = json.loads(capturedContent)

// execute command['name'](**command['arguments'])

}

data = [
    {
        'input': ....
        'output': ...
        'comamndName': ..Value
        'task': ....
    },
    {
        'input': ....
        'output': ...
        'comamndName': ..Value
        'task': ....
    },
]

parquet = pd.Dataframe(data)
parquet.to_paruet("./file.parquet')