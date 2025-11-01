import React from "react";
// It renders a single chat message (either from the user or the AI) in your chat interface.
interface MessageProps {
  sender: string;
  text: string;
}

const Chat: React.FC<MessageProps> = ({ sender, text }) => {
  const isUser = sender.toLowerCase() === "user";
  return (
    <div className={`row ${isUser ? "right" : "left"}`}>
      <div className={`bubble ${isUser ? "user" : "ai"}`}>{text}</div>
    </div>
  );
};

export default Chat;
