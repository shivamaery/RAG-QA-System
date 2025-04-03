import React from "react";
// It renders a single chat message (either from the user or the AI) in your chat interface.
interface MessageProps {
  sender: string;
  text: string;
}

const Chat: React.FC<MessageProps> = ({ sender, text }) => {
  return (
    <p>
      <strong>{sender}:</strong> {text}
    </p>
  );
};

export default Chat;
