import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import styled from 'styled-components'
import { Streamlit } from "streamlit-component-lib"

const SliderContainer = styled.div`
  width: 100%;
  padding: 20px 0;
  position: relative;
`

const Track = styled.div`
  width: 100%;
  height: 2px;
  background: rgba(255, 255, 255, 0.2);
  position: relative;
`

const Progress = styled.div`
  height: 100%;
  background: #0066cc;
  width: ${props => props.percentage}%;
  position: absolute;
  left: 0;
  top: 0;
`

const Handle = styled.div`
  width: 24px;
  height: 24px;
  background: white;
  border-radius: 50%;
  position: absolute;
  top: 50%;
  left: ${props => props.percentage}%;
  transform: translate(-50%, -50%);
  cursor: pointer;
  transition: transform 0.1s ease;

  &:hover {
    transform: translate(-50%, -50%) scale(1.1);
  }
`

const Labels = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 10px;
  position: relative;
  padding: 0 12px;
`

const Label = styled.span`
  color: ${props => props.active ? '#0066cc' : '#fff'};
  font-family: "Source Sans Pro", sans-serif;
  font-size: 14px;
  cursor: pointer;
  transition: color 0.2s ease;

  &:hover {
    color: #0066cc;
  }
`

const DepthSlider = ({ options, value, setComponentValue }) => {
  const [currentValue, setCurrentValue] = useState(value)
  const currentIndex = options.indexOf(currentValue)
  const percentage = (currentIndex / (options.length - 1)) * 100

  const handleClick = (option) => {
    setCurrentValue(option)
    setComponentValue(option)
  }

  useEffect(() => {
    if (value !== currentValue) {
      setCurrentValue(value)
    }
  }, [value, currentValue])

  return (
    <SliderContainer>
      <Track>
        <Progress percentage={percentage} />
        <Handle percentage={percentage} />
      </Track>
      <Labels>
        {options.map((option) => (
          <Label
            key={option}
            active={option === currentValue}
            onClick={() => handleClick(option)}
          >
            {option}
          </Label>
        ))}
      </Labels>
    </SliderContainer>
  )
}

const StreamlitDepthSlider = () => {
  const [componentValue, setComponentValue] = useState(null)

  useEffect(() => {
    Streamlit.setComponentReady()
  }, [])

  useEffect(() => {
    if (componentValue !== null) {
      Streamlit.setComponentValue(componentValue)
    }
  }, [componentValue])

  const args = Streamlit.args || { options: [], value: "" }

  return (
    <DepthSlider
      options={args.options || []}
      value={args.value || ""}
      setComponentValue={setComponentValue}
    />
  )
}

ReactDOM.render(
  <React.StrictMode>
    <StreamlitDepthSlider />
  </React.StrictMode>,
  document.getElementById('root')
) 