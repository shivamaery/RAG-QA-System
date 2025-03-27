import React, { useState } from "react";

function InputBox(): JSX.Element {
  const [inputVal, setInputVal] = useState<string>("");

  //If empty input is returned, do nothing
  const handleEnter = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (inputVal.trim() === "") return;
    console.log(inputVal);
  };

  return (
    <div className="InputBox">
      <form onSubmit={handleEnter}>
        <input
          type="text"
          value={inputVal}
          onChange={(e) => setInputVal(e.target.value)}
          placeholder="Type something..."
        />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
}
export default InputBox;
