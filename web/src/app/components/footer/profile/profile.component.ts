import { Component } from '@angular/core';
import { MatDialogRef } from '@angular/material';
import { Name, Profile } from './profiles';


@Component({
  templateUrl: './profile.component.html'
})
export class ProfileComponent {
  public profile: Profile;

  constructor(
    public dialogRef: MatDialogRef<ProfileComponent>
  ) {

  }
}
