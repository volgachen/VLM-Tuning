# Conversation

## Variables

- **system** (str) The system prompt that appears at the start of the conversation
- **roles** (List[str]) A list (often two) of all possible roles in the 
- **sep_style** (SeparatorStyle)
- **sep** (str) separator after messages. If sep2 is provided, it only after human messages
- **sep2** (str) only after AI messages
- **offset**
- **messages**

## Methods
- **append_message**
- **get_prompt**
- **dict**

## SeparatorStyle

- **SINGLE** {SYSTEM}<sep>{ROLE}: {MSG}<sep>[{ROLE}:]
- **TWO** {SYSTEM}<sep>{ROLE[0]}: {MSG}<sep>{ROLE[1]}: {MSG}<sep2>
- **PLAIN** {MSG}<sep>{MSG}<sep2> 