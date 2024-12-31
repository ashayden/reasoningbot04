import React, { useEffect, useState, useRef } from 'react'
import ReactDOM from 'react-dom'
import styled from 'styled-components'
import {
  Streamlit,
  withStreamlitConnection,
} from "streamlit-component-lib"

const SliderContainer = styled.div`
  width: 100%;
  padding: 40px 12px 20px;
  position: relative;
  user-select: none;
`

const Label = styled.div`
  font-family: "Source Sans Pro", sans-serif;
  font-size: 14px;
  color: #fff;
  margin-bottom: 16px;
`

const Track = styled.div`
  width: 100%;
  height: 2px;
  background: rgba(255, 255, 255, 0.2);
  position: relative;
  cursor: pointer;
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
  cursor: grab;
  transition: transform 0.1s ease;
  z-index: 1;

  &:hover {
    transform: translate(-50%, -50%) scale(1.1);
  }

  &:active {
    cursor: grabbing;
    transform: translate(-50%, -50%) scale(1.1);
  }
`

const Labels = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 16px;
  position: relative;
`

const Option = styled.span`
  color: ${props => props.active ? '#0066cc' : '#fff'};
  font-family: "Source Sans Pro", sans-serif;
  font-size: 14px;
  cursor: pointer;
  transition: color 0.2s ease;
  padding: 4px;

  &:hover {
    color: #0066cc;
  }
`

const DepthSlider = ({ args, disabled }) => {
  const { options = [], value = "", label = "Analysis Depth" } = args
  const [currentValue, setCurrentValue] = useState(value)
  const [isDragging, setIsDragging] = useState(false)
  const trackRef = useRef(null)
  
  const currentIndex = options.indexOf(currentValue)
  const percentage = (currentIndex / (Math.max(options.length - 1, 1))) * 100

  useEffect(() => {
    // Initialize the component
    Streamlit.setFrameHeight()
  }, [])

  useEffect(() => {
    // Update value when props change
    if (value !== currentValue) {
      setCurrentValue(value)
    }
  }, [value])

  const getValueFromPosition = (clientX) => {
    if (!trackRef.current) return currentValue
    
    const rect = trackRef.current.getBoundingClientRect()
    const position = (clientX - rect.left) / rect.width
    const index = Math.round(position * (options.length - 1))
    return options[Math.max(0, Math.min(index, options.length - 1))]
  }

  const handleMouseDown = (e) => {
    if (disabled) return
    setIsDragging(true)
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  const handleMouseMove = (e) => {
    if (!isDragging) return
    const newValue = getValueFromPosition(e.clientX)
    if (newValue !== currentValue) {
      setCurrentValue(newValue)
      Streamlit.setComponentValue(newValue)
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
  }

  const handleTrackClick = (e) => {
    if (disabled) return
    const newValue = getValueFromPosition(e.clientX)
    setCurrentValue(newValue)
    Streamlit.setComponentValue(newValue)
  }

  const handleOptionClick = (option) => {
    if (disabled) return
    setCurrentValue(option)
    Streamlit.setComponentValue(option)
  }

  return (
    <SliderContainer>
      <Label>{label}</Label>
      <Track ref={trackRef} onClick={handleTrackClick}>
        <Progress percentage={percentage} />
        <Handle 
          percentage={percentage}
          onMouseDown={handleMouseDown}
        />
      </Track>
      <Labels>
        {options.map((option) => (
          <Option
            key={option}
            active={option === currentValue}
            onClick={() => handleOptionClick(option)}
          >
            {option}
          </Option>
        ))}
      </Labels>
    </SliderContainer>
  )
}

// Wrap the component with Streamlit's connection HOC
const StreamlitDepthSlider = withStreamlitConnection(DepthSlider)

// Render with strict mode
ReactDOM.render(
  <React.StrictMode>
    <StreamlitDepthSlider />
  </React.StrictMode>,
  document.getElementById('root')
) 