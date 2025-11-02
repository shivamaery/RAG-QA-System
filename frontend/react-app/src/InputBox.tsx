import React, { useState } from "react";
import Chat from "./Chat";

function InputBox(): JSX.Element {
  const [inputVal, setInputVal] = useState<string>("");
  const [chat, setChat] = useState<{ sender: string; text: string }[]>([]);

  const getFakeAIResponse = (userInput: string): string => {
    return `You said: "${userInput}". `;
  };

  //If empty input is returned, do nothing
  const handleEnter = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (inputVal.trim() === "") return;
    console.log(inputVal);

    setChat((prevChat) => [...prevChat, { sender: "user", text: inputVal }]);
    setInputVal("");

    const aiMsg = { sender: "AI", text: getFakeAIResponse(inputVal) };
    setChat((prevChat) => [...prevChat, aiMsg]);
  };

  return (
    <div>
      <h1 className="header">ResearchAI</h1>

      <div className="chat-window">
        {chat.map((msg, i) => (
          <Chat key={i} sender={msg.sender} text={msg.text} />
        ))}
      </div>

      <form onSubmit={handleEnter}>
        <input
          className="input-field"
          type="text"
          value={inputVal}
          onChange={(e) => setInputVal(e.target.value)}
          placeholder=" How can I help you?"
        />
        <button type="submit" className="submit-btn">
          &#x2794;
        </button>
      </form>
    </div>
  );
}

export default InputBox;
