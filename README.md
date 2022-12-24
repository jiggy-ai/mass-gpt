<p align="center">
<img src="https://github.com/jiggy-ai/mass-gpt/blob/main/MassGPT.jpg" alt="MassGPTBot" width=256> 
</p>

**MassGPT** (https://t.me/MassGPTbot) is an open source Telegram bot that interacts with users by responding to messages via a GPT3.5 (davinci-003) completion that is conditioned by a shared sub-prompt context of all recent user messages.

Message the MassGPT Telegram bot directly, and it will respond back to you taking into account the content of relevant messages it has received from others.  Users are assigned a numeric user id that does not leak any Telegram user info.  The messages you send are not private as they are accessible to other users via the bot or via the /context command.  

You can also send the bot a url to inject a summary of the web page text into the current context.  Use the /context command to see the entire current context. 

Currently there is one global chat context of recent messages from other users, but I plan to scale this by using an embedding of the user message to dynamically assemble a relevant context via ANN query of past messages, message summaries, and url summaries.

There are several motivations for creating this.  One is to explore chatting with an LLM where the mode is presenting some partial consensus of other user’s current inputs instead of the model’s background representations.  Another is to explore dynamic prompt contexts which should have a lot of interesting applications given that GPT seems much less likely to hallucinate when summarizing from its current prompt context.

Open to PRs if you are interested in hacking on this in any way, or feel free to message me @ wskish on Twitter or Telegram.  



**OpenAI**

* OPENAI_API_KEY # your OpenAI API key


**PostgresQL** 

Database for keeping track of items we have already seen and associated item info.

- MASSGPT_POSTGRES_HOST  # The database FQDN
- MASSGPT_POSTGRES_USER  # The database username
- MASSGPT_POSTGRES_PASS  # The database password

**Telegram**
  
* MASSGPT_TELEGRAM_API_TOKEN # The bot's telegram API token



