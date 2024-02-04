# Llamacron: "When time gets right"
![image](https://github.com/tslmy/agent/assets/594058/f9260527-de8a-47ac-bf73-573301fcc17e)

_A unicorn llama, generated via MidJourney_

Interruption can be productive.

## Problem statement

What's missing?

### Humans can postpone tasks

![](https://github.com/tslmy/agent/assets/594058/4ea77ab2-77eb-4e21-a400-1e95b52dcb27)

([source](https://www.pexels.com/photo/a-small-dog-standing-in-front-of-a-window-looking-out-16479077/))

Watching a movie together in a cozy afternoon, you asked your partner to walk the dog.

They looked out the window, saying, "It's raining outside. Maybe later", and rejoined you on the couch.

When the sun came out, **without you asking again** (if you had to, reconsider your marriage), they said, "The time is right. I'll walk the dog now."

### Few AIs do that today

If you married an AI (I'm not judging), they will:
- either outright refuse to walk the dog and completely forget about it,
- or start staring at the window for 5 hours, ruining the better half of the Netflix marathon.

They lack a sense of "back burner".

![Miele KM391GBL 36" Black Gas Sealed Burner Cooktop](https://github.com/tslmy/agent/assets/594058/ef53719a-827a-4cdc-a8d1-1e3fcaa56488)

## Solution

Implement a tool that, when the last step in the chain of thought (CoT) deemed it isn't yet the right time to do something, spin off a thread to check the precondition periodically. Don't block the chat.

When the precondition is met, we resume the task. Append the task result to the chat history, as if the LLM had just said, "Hey, by the way, the sky cleared up, so I walked Fido and he's so happy."

## Demo

Ask the AI, **"Please go walk the dog."** It will say "It's raining; maybe later".

Continue the conversation by talking about something else. Perhaps **"how are you feeling right now"**. The chatbot will follow the flow.

Soon, the AI will attempt to walk the dog again, and sees that the sky has cleared up, so it will say, "I walked the dog, and he really enjoyed the park."

It's not just a UI trick. You can ask, **"Can you rephrase that?"**. The AI is aware of how the conversation has diverged.

## Future work

**The condition can become true as the conversation evolves.** (“Hey, did you just say Z is true? You know what, that actually implies that Y is true, so I’ll go ahead and do X now.“)

This means a traditional, static cron job won’t cut it. The AI has to somehow update the context of each desired action X that was held off.

**Humans know when to give up.** If the precondition turned out to be impossible to come true, remove X from its back burner.

- "Throw me a party when I marry Taylor Swift",
- "Remind me to [kill Sarah Connor](https://en.wikipedia.org/wiki/The_Terminator) when we get back to 1984",
- ...

“Dang it! Now that we realized that Y will never be the case, let’s forget about doing X for good.”
