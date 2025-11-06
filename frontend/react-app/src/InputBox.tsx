import React, { useState } from "react";
import Chat from "./Chat";
import axios from "axios";

function InputBox(): JSX.Element {
  const [inputVal, setInputVal] = useState<string>("");
  const [chat, setChat] = useState<{ sender: string; text: string }[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const handleEnter = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (inputVal.trim() === "") return;

    const userMsg = { sender: "user", text: inputVal };
    setChat((prevChat) => [...prevChat, userMsg]);
    setInputVal("");
    setLoading(true);

    try {
      const response = await axios.post("http://127.0.0.1:8000/query", {
        question: inputVal,
      });

      const aiResponse = response.data.answer;
      const sources = response.data.sources;

      const aiMsg = {
        sender: "AI",
        text: `${aiResponse}\n\nSources: ${sources.join(", ") || "none"}`,
      };

      setChat((prevChat) => [...prevChat, aiMsg]);
    } catch (error: any) {
      console.error("Error fetching AI response:", error);
      setChat((prevChat) => [
        ...prevChat,
        { sender: "AI", text: "Error: could not get a response from backend." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="chat-window">
        {chat.map((msg, i) => (
          <Chat key={i} sender={msg.sender} text={msg.text} />
        ))}
        {loading && <div className="loading">Thinking...</div>}
      </div>

      <form onSubmit={handleEnter}>
        <input
          className="input-field"
          type="text"
          value={inputVal}
          onChange={(e) => setInputVal(e.target.value)}
          placeholder=" How can I help you?"
        />
        <button type="submit" className="submit-btn" disabled={loading}>
          &#x2794;
        </button>
      </form>
    </div>
  );
}

export default InputBox;
