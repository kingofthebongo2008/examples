//-----------------------------------------------------------------------------
// File: Framework\GUI.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------



#include "GUI.h"

#if 0

Slider::Slider(const float ix, const float iy, const float w, const float h, const bool horizontal, const float ir0, const float ir1, const float val, const float stepSize){
	drag = false;
	xPos = ix;
	yPos = iy;
	width  = w;
	height = h;
	isHorizontal = horizontal;
	step = stepSize;
	setRange(ir0, ir1);
	setValue(val);
	color = vec4(1, 1, 0, 0.5f);
	sliderListener = NULL;
}

void Slider::draw(Renderer *renderer){
#ifdef D3D10


#else
	renderer->setDepthFunc(DEPTH_NONE);
	renderer->setBlending(SRC_ALPHA, ONE_MINUS_SRC_ALPHA);
	renderer->apply();

	// Background
	renderer->drawRoundRect(xPos, yPos, xPos + width, yPos + height, 10, color);

	renderer->setDepthFunc(DEPTH_NONE);
	renderer->apply();

	vec4 black(0, 0, 0, 1);
	// Border
	renderer->drawRoundRect(xPos, yPos, xPos + width, yPos + height, 10, black, 3);

	if (isHorizontal){
		float vx = lerp(xPos + 12.0f, xPos + width - 12.0f, (value - r0) / (r1 - r0)) + 0.5f;
		renderer->drawLine(xPos + 12, yPos + 0.5f * height, xPos + width - 12, yPos + 0.5f * height, black, 3);
		renderer->drawRoundRect(vx - 5, yPos + 3, vx + 5, yPos + height - 3, 4, black);
	} else {
		float vy = lerp(yPos + 12.0f, yPos + height - 12.0f, (r1 - value) / (r1 - r0)) + 0.5f;
		renderer->drawLine(xPos + 0.5f * width, yPos + 12, xPos + 0.5f * width, yPos + height - 12, black, 3);
		renderer->drawRoundRect(xPos + 3, vy - 5, xPos + width - 3, vy + 5, 4, black);
	}
#endif
}

bool Slider::isInWidget(const int x, const int y) const {
	return (x >= xPos && x < xPos + width && y >= yPos && y < yPos + height);
}

bool Slider::onMouseClick(const int x, const int y, const unsigned int button, const bool pressed){
	if (button == LEFT_BUTTON){
		drag = capture = pressed;
		if (pressed) computeValue(x, y);
		return true;
	}
	return false;
}

bool Slider::onMouseMove(const int x, const int y, const bool lButton, const bool mButton, const bool rButton){
	if (drag){
		computeValue(x, y);
		return true;
	}
	return false;
}

void Slider::setValue(const float val){
	value = clamp(val, r0, r1);
	if (step > 0){
		float s = (value - r0) / step;
		s = floor(s + 0.5f);
		value = min(r0 + s * step, r1);
	}
}

void Slider::computeValue(const int x, const int y){
	float k = isHorizontal? (x - (xPos + 12.0f)) / (width - 24.0f) : ((yPos + height - 12.0f) - y) / (height - 24.0f);
	setValue(lerp(r0, r1, k));
	if (sliderListener) sliderListener->onSliderChange(this);
}

#endif