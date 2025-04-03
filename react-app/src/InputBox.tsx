import React, { useState } from "react";
import Chat from "./Chat";

function InputBox(): JSX.Element {
  const [inputVal, setInputVal] = useState<string>("");
  const [chat, setChat] = useState<{ sender: string; text: string }[]>([]);

  //If empty input is returned, do nothing
  const handleEnter = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (inputVal.trim() === "") return;
    console.log(inputVal);

    setChat((prevChat) => [
      ...prevChat,
      { sender: "user", text: inputVal },
    ]);
    setInputVal("");

  };

 


  return (
    <div>
      <h1>ResearchAI</h1>
        {chat.map((msg, index) => (
          <Chat key={index} sender={msg.sender} text={msg.text} />
        ))}
  

    
      <form onSubmit={handleEnter}>
        <input
          className="input-field"
          type="text"
          value={inputVal}
          onChange={(e) => setInputVal(e.target.value)}
          placeholder=" How can I help you?"
        />
        <button type="submit" className="submit-btn">
          ^
        </button>
      </form>
    </div>
    
  );
}

export default InputBox;
