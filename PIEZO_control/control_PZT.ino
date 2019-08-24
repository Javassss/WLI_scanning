// MCPDAC relies on SPI.
#include <SPI.h>
#include "MCPDAC.h"

String incr;
String steps;
unsigned int num_steps;
unsigned int volt_incr;
int ctr;

typedef enum e_State {
  STATE_READ_STEPS, // wait for number of steps from the serial
  STATE_READ_INCR, // wait for voltage increment from the serial
  STATE_PREPARE,
  STATE_MOVE_PZT,  // moving PZT
  STATE_WAIT_ACK   // waiting acknowledge from message
} eState;

void setup()
{
  Serial.begin(9600);
  
  // CS on pin 10, no LDAC pin (tie it to ground).
  MCPDAC.begin(10);
  
  // Set the gain to "HIGH" mode - 0 to 4096mV.
  MCPDAC.setGain(CHANNEL_A,GAIN_HIGH);
  
  // Do not shut down channel A, but shut down channel B.
  MCPDAC.shutdown(CHANNEL_A,false);
  MCPDAC.shutdown(CHANNEL_B,true);
}

void loop()
{
  static unsigned int volts = 4096;
  static int max_strlen = 5;
  static int ascend = 0;
  static int stp = 0;
  static eState LoopState = STATE_READ_STEPS; // to store the current state
  eState NextState = LoopState;
  static char a = 'a'; static char d = 'd'; char s;

  switch (LoopState) {
    case STATE_READ_STEPS:
      if (Serial.available() == max_strlen) {
        // get the number of steps from the serial
        steps = Serial.readStringUntil('\n');
        num_steps = decode_string2int(steps);
        NextState = STATE_READ_INCR;
        break;
      }else{
        NextState = STATE_READ_STEPS;
        break;
      }
    case STATE_READ_INCR:
      if (Serial.available() == max_strlen) {
        ctr = 0;
        // get the voltage increment from the serial
        incr = Serial.readStringUntil('\n');
        volt_incr = decode_string2int(incr);
        NextState = STATE_PREPARE;
        break;
      }else{
        NextState = STATE_READ_INCR;
        break;
      }
    case STATE_PREPARE:
      volts = 4096;
      MCPDAC.setVoltage(CHANNEL_A,volts&0x0fff);
      delay(5000);
      NextState = STATE_MOVE_PZT;
      break;
    case STATE_MOVE_PZT:
      ctr++;
      // Set the voltage of channel A.
      MCPDAC.setVoltage(CHANNEL_A,volts&0x0fff);
      delay(10);
      s = (ascend == 1) ? a : d ;
      Serial.print(s);
      Serial.print("_");
      Serial.println(volts); // send signal to python
      Serial.flush(); // wait for data to be transmitted
      NextState = STATE_WAIT_ACK;
      break;
    case STATE_WAIT_ACK:
      if (Serial.available() == 3 && (Serial.readStringUntil('\n')).equals("ok")) { 
        // no state change while no acknowledge
        unsigned int volt_plus = volts + volt_incr;
        unsigned int volt_minus = volts - volt_incr;

        if (ascend == 1){
          if ( (volt_plus >=0) && (volt_plus <= 4096) ){
            volts += volt_incr;
          }else{
            ascend = 0;
            volts -= volt_incr;
          }
        }else{
          if ( (volt_minus >=0) && (volt_minus <= 4096) ){
            volts -= volt_incr;
          }else{
            ascend = 1;
            volts += volt_incr;
          }
        }
        NextState = STATE_MOVE_PZT;
        break;
      }else if (num_steps+1 == ctr) {
        volts = 0;
        MCPDAC.setVoltage(CHANNEL_A,volts&0x0fff);
        NextState = STATE_READ_STEPS;
        break;
      }else{
        NextState = STATE_WAIT_ACK;
        break;
      }
  }
  
  // define the state for the next loop
  LoopState = NextState;
}

// discard possible zeros from the input string and convert the string to integer
int decode_string2int(String str)
{
  while (str[0] == '0') {
    str.remove(0,1);
  }
  return str.toInt();
}
