//-----------------------------------------------------------------------------
// File: Framework\GUI.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------



#ifndef _GUI_H_
#define _GUI_H_

#include "D3D11/D3D11Context.h"
#include "Font.h"

class Widget
{
public:
	Widget(){
		m_visible = true;
		m_capture = false;
		m_dead = false;
	}
	virtual ~Widget(){}

	virtual void Render() = 0;
	virtual bool IsInWidget(const int x, const int y) const = 0;
	virtual bool OnMouseClick(const int x, const int y, const unsigned int button, const bool pressed){ return false; }
	virtual bool OnMouseMove (const int x, const int y, const bool lButton, const bool mButton, const bool rButton){ return false; }
	virtual bool OnMouseWheel(const int x, const int y, const int scroll){ return false; }
	virtual bool OnKeyPress(const unsigned int key, const bool pressed){ return false; }
	virtual void OnSize(const int w, const int h, const int oldW, const int oldH){}

	void SetColor(const float4 &col){ m_color = col; }
	virtual void SetPosition(const float newX, const float newY);
	void SetVisible(const bool isVisible){ m_visible = isVisible; }
	bool IsVisible() const { return m_visible; }
	bool IsCapturing() const { return m_capture; }
	bool IsDead() const { return m_dead; }

protected:
	float4 m_color;
	float m_xPos, m_yPos;
	bool m_visible;
	bool m_capture;
	bool m_dead;
};

//---------------------------------------------------------------------------------------------

class Button : public Widget
{
public:


};

//---------------------------------------------------------------------------------------------

#if 0

class Slider;

/** A slider listener listens for framework callbacks on slider interaction */
class SliderListener {
public:
	virtual void onSliderChange(Slider *slider) = 0;
};

/** A basic slider bar widget */
class Slider : public Widget {
public:
	Slider(const float ix, const float iy, const float w, const float h, const bool horizontal = true, const float ir0 = 0, const float ir1 = 1, const float val = 0, const float stepSize = 0);

	void draw(Renderer *renderer);
	bool isInWidget(const int x, const int y) const;
	bool onMouseClick(const int x, const int y, const unsigned int button, const bool pressed);
	bool onMouseMove (const int x, const int y, const bool lButton, const bool mButton, const bool rButton);

	void setRange(const float ir0, const float ir1){
		r0 = ir0;
		r1 = ir1;
	}
	void setSize(const float w, const float h){
		width  = w;
		height = h;
	}

	float getValue() const { return value; }
	void setValue(const float val);
	void setListener(SliderListener *listener){ sliderListener = listener; }

protected:
	void computeValue(const int x, const int y);

	float width, height;

	float r0, r1;
	float value;
	float step;

	SliderListener *sliderListener;

	bool isHorizontal;
	bool drag;
};

#endif
//---------------------------------------------------------------------------------------------

class GUI
{
public:
	GUI();
	~GUI();

	Button *AddButton(const float x, const float y, const float width, const float height, const char *text, const float textSize);

	void Render();
protected:


};

#endif // _GUI_H_
